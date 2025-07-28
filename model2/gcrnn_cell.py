import torch
import torch.nn as nn
import numpy as np
import torch

from lib import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        """
        作用：定义属性，初始化参数和偏置
        """
        self._rnn_network = rnn_network # rnn网络
        self._params_dict = {} # 存放参数的字典
        self._biases_dict = {} # 存放偏置的字典
        self._type = layer_type # 每层的类型

    def get_weights(self, shape):
        """
        作用：根据给出的shape，初始化权重参数
        """
        if shape not in self._params_dict:
            # torch.nn.Parameter():是一个Tensor，也就是说Tensor 拥有的属性它都有
            # 首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter；
            # 并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))
            torch.nn.init.xavier_normal_(nn_param) # 前面把nn_param用0填充，现在再用正态分布的方式初始化
            self._params_dict[shape] = nn_param # 加入到参数字典中

            # .register_parameter()作用和.Parameter()一样，只不过是向 我们建立的网络module添加 parameter
            # 第一个参数为参数名字，第二个参数为Parameter()对象，其实是个Tensor矩阵
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        """
        作用：根据长度 初始化偏置
        """
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start) # 用值bias_start填充向量biases。
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]

class GCLSTMCell(torch.nn.Module):
    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, nonlinearity='tanh', filter_type="laplacian"):
        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self.adj_mx = adj_mx
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        supports = []

        # 选择合适的图卷积过滤器
        if filter_type == "laplacian":
            supports.append(utils.calculate_scaled_laplacian(self.adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(self.adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(utils.calculate_random_walk_matrix(self.adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(self.adj_mx.T).T)
        else:
            supports.append(utils.calculate_scaled_laplacian(self.adj_mx))

        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))

        self._fc_params = LayerParams(self, 'fc')
        self._gconv_params = LayerParams(self, 'gconv')

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L

    @staticmethod
    def cheb_polynomial(laplacian, K):
        """
        计算切比雪夫多项式。
        """
        N = laplacian.size(0)
        multi_order_laplacian = torch.zeros([K, N, N], device=device)
        multi_order_laplacian[0] = torch.eye(N, device=device)

        if K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            for k in range(2, K):
                multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k - 1]) - multi_order_laplacian[k - 2]
        return multi_order_laplacian

    @staticmethod
    def get_laplacian(adj):
        """
        返回图的拉普拉斯矩阵。
        """
        D = torch.diag(torch.sum(adj, dim=-1) ** (-1 / 2))
        D = torch.where(torch.isinf(D), torch.full_like(D, 0), D)
        L = torch.eye(adj.size(0), device=adj.device) - torch.mm(torch.mm(D, adj), D)
        return L

    def forward(self, inputs, hx, cx):
        # print(f"inputs shape: {inputs.shape}")
        # print(f"hx shape: {hx.shape}")
        # print(f"cx shape: {cx.shape}")

        output_size = 4 * self._num_units  # LSTM 有4个输出（输入门、遗忘门、输出门、候选记忆）

        # 图卷积操作
        gconv_output = self._gconv(inputs, hx, output_size)
        gconv_output = torch.reshape(gconv_output, (-1, self._num_nodes, output_size))

        # 切分图卷积输出为 i, f, o, g
        i, f, o, g = torch.split(gconv_output, self._num_units, dim=-1)

        i = torch.sigmoid(i)  # 输入门
        f = torch.sigmoid(f)  # 遗忘门
        o = torch.sigmoid(o)  # 输出门
        g = torch.tanh(g)     # 候选记忆

        # 将 cx reshape 为与 f, i, g 相同的维度
        cx = torch.reshape(cx, (-1, self._num_nodes, self._num_units))
        # print(f"cx (after reshape) shape: {cx.shape}")

        # 更新记忆单元和隐藏状态
        new_cx = f * cx + i * g  # 新的记忆单元
        new_hx = o * torch.tanh(new_cx)  # 新的隐藏状态

        return new_hx, new_cx  # 返回新的隐藏状态和记忆单元

    def _gconv(self, inputs, state, output_size):
        """
        图卷积操作：结合图的邻接矩阵处理节点之间的依赖关系
        :param inputs: (batch_size, num_nodes * input_dim)
        :param state: (batch_size, num_nodes * rnn_units)
        :param output_size: 输出尺寸
        :return: 图卷积后的结果
        """
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)  # (batch_size, num_nodes, input_dim + rnn_units)
        input_size = inputs_and_state.shape[-1]

        x0 = inputs_and_state.permute(1, 2, 0).reshape(self._num_nodes, input_size * batch_size)
        x = torch.unsqueeze(x0, 0)

        for support in self._supports:
            x1 = torch.sparse.mm(support, x0)
            x = self._concat(x, x1)

        num_matrices = len(self._supports) + 1
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0).reshape(batch_size * self._num_nodes, input_size * num_matrices)

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)
        biases = self._gconv_params.get_biases(output_size)
        x += biases

        return torch.reshape(x, [batch_size, self._num_nodes * output_size])

    def _chebconv(self, inputs, state, output_size, bias_start=0.0):
        """
        切比雪夫多项式卷积操作。
        """
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x0 = inputs_and_state.permute(1, 2, 0).reshape(self._num_nodes, input_size * batch_size)
        x = torch.unsqueeze(x0, 0)

        L = self.get_laplacian(torch.tensor(self.adj_mx, device=device))
        mul_L = self.cheb_polynomial(L, K=2)

        for _ in range(len(self._supports)):
            x1 = torch.matmul(mul_L, x0)
            x1 = torch.sum(x1, dim=0)
            x = self._concat(x, x1)

        num_matrices = len(self._supports) + 1
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0).reshape(batch_size * self._num_nodes, input_size * num_matrices)

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases

        return torch.reshape(x, [batch_size, self._num_nodes * output_size])

    def _dconv(self, inputs, state, output_size, bias_start=0.0):
        """
        扩散卷积操作。
        """
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0).reshape(self._num_nodes, input_size * batch_size)
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step > 0:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._max_diffusion_step + 1
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0).reshape(batch_size * self._num_nodes, input_size * num_matrices)

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases

        return torch.reshape(x, [batch_size, self._num_nodes * output_size])

    def _gatconv(self, inputs, state, output_size):
        # GAT卷积操作：使用图注意力机制处理节点之间的依赖关系
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)  # (batch_size, num_nodes, input_dim + rnn_units)
        input_size = inputs_and_state.size(2)

        # 初始化权重矩阵
        x = inputs_and_state

        # 对每一个注意力头进行处理
        multi_head_outputs = []
        for _ in range(self._n_heads):
            weights = self._gconv_params.get_weights((input_size, output_size))
            b = self._gconv_params.get_biases(output_size)
            h = torch.matmul(x, weights)  # (batch_size, num_nodes, output_size)

            # 计算注意力系数
            attention_scores = torch.bmm(h, h.transpose(1, 2))  # (batch_size, num_nodes, num_nodes)
            attention_scores = attention_scores * self.adj_mx.unsqueeze(0)  # 使用邻接矩阵掩码
            attention_scores.data.masked_fill_(torch.eq(attention_scores, 0), -float(1e16))  # 掩盖无连接的边

            # softmax 归一化
            attention_weights = torch.softmax(attention_scores, dim=2)

            # 使用注意力权重进行加权求和
            attention_output = torch.bmm(attention_weights, h) + b  # (batch_size, num_nodes, output_size)
            multi_head_outputs.append(attention_output)

        # 将多个头的输出拼接
        multi_head_output = torch.cat(multi_head_outputs, dim=-1)  # (batch_size, num_nodes, output_size * n_heads)

        return torch.reshape(multi_head_output, [batch_size, self._num_nodes * output_size * self._n_heads])

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)