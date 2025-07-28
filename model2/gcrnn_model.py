import torch
import numpy as np
import torch.nn as nn

from model2.gcrnn_cell import GCLSTMCell

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    """
    Gets the total number of trainable parameters of the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # p.numel 意思为：p中元素的数量

class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx  # 邻接矩阵
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))  # 扩散卷积中 最大的扩散步数
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')  # 滤波器的类型
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))  # 节点的数量
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))  # rnn的层数
        self.rnn_units = int(model_kwargs.get('rnn_units'))  # units 就是 cell 中输出层的维度
        self.hidden_state_size = self.num_nodes * self.rnn_units  # 隐藏层的大小

# Attention mechanism for time steps
class SimpleAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        encoder_outputs: (seq_len, batch_size, num_nodes, hidden_size)
        decoder_hidden: (batch_size, num_nodes, hidden_size)
        """
        seq_len = encoder_outputs.size(0)  # 12个时间步
        batch_size = encoder_outputs.size(1)
        num_nodes = encoder_outputs.size(2)
        #
        # print(f"Encoder outputs shape: {encoder_outputs.shape}")
        # print(f"Decoder hidden shape: {decoder_hidden.shape}")

        # 扩展解码器的隐藏状态
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1, 1)  # (batch_size, seq_len, num_nodes, hidden_size)
        # print(f"Decoder hidden after unsqueeze and repeat shape: {decoder_hidden.shape}")

        # 计算注意力权重
        energy = torch.tanh(self.attn(encoder_outputs))  # (seq_len, batch_size, num_nodes, hidden_size)
        energy = torch.sum(energy * decoder_hidden.permute(1, 0, 2, 3), dim=-1)  # (seq_len, batch_size, num_nodes)

        # 计算注意力权重 (batch_size, num_nodes, seq_len)
        attention_weights = torch.softmax(energy.permute(1, 2, 0), dim=-1)  # (batch_size, num_nodes, seq_len)
        # print(f"Attention weights shape: {attention_weights.shape}")

        return attention_weights

class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)

        self.input_dim = int(model_kwargs.get('input_dim', 11))
        self.seq_len = int(model_kwargs.get('seq_len'))

        # 替换为 GCLSTMCell
        self.gclstm_layers = nn.ModuleList(
            [GCLSTMCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                        filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None, cell_state=None):
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size), device=device)
        if cell_state is None:
            cell_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size), device=device)

        hidden_states = []
        cell_states = []
        output = inputs
        for layer_num, gclstm_layer in enumerate(self.gclstm_layers):
            next_hidden_state, next_cell_state = gclstm_layer(output, hidden_state[layer_num], cell_state[layer_num])
            hidden_states.append(next_hidden_state)
            cell_states.append(next_cell_state)
            output = next_hidden_state

        return output, (torch.stack(hidden_states), torch.stack(cell_states))

class AttentionDecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)

        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 3))  # 预测未来3个时间步
        self.projection_layer = nn.Linear(self.rnn_units * 2, self.output_dim)
        self.attention_layer = SimpleAttentionLayer(self.rnn_units)

        self.gclstm_layers = nn.ModuleList(
            [GCLSTMCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                        filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, encoder_outputs, hidden_state=None, cell_state=None):
        hidden_states = []
        cell_states = []
        output = inputs
        for layer_num, gclstm_layer in enumerate(self.gclstm_layers):
            next_hidden_state, next_cell_state = gclstm_layer(output, hidden_state[layer_num], cell_state[layer_num])
            hidden_states.append(next_hidden_state)
            cell_states.append(next_cell_state)
            output = next_hidden_state

        # print(f"Decoder output shape (before attention): {output.shape}")

        # 计算注意力权重
        attention_weights = self.attention_layer(encoder_outputs, output)

        # 基于注意力权重加权求和，生成上下文向量
        context_vector = torch.bmm(attention_weights.view(-1, 1, encoder_outputs.size(0)),
                                   encoder_outputs.permute(1, 2, 0, 3).contiguous().view(-1, encoder_outputs.size(0), self.rnn_units)).squeeze(1)
        # print(f"Context vector shape: {context_vector.shape}")

        # 将上下文向量与当前隐藏状态拼接
        output = torch.cat((output, context_vector.view(output.size(0), output.size(1), -1)), dim=-1)

        projected = self.projection_layer(output.view(-1, self.rnn_units * 2))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, (torch.stack(hidden_states), torch.stack(cell_states)), attention_weights

class GCRNNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, logger, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)

        self.encoder_model = EncoderModel(adj_mx, **model_kwargs)
        self.decoder_model = AttentionDecoderModel(adj_mx, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        encoder_hidden_state, encoder_cell_state = None, None
        encoder_outputs = []
        for t in range(self.encoder_model.seq_len):
            encoder_output, (encoder_hidden_state, encoder_cell_state) = self.encoder_model(inputs[t], encoder_hidden_state, encoder_cell_state)
            encoder_outputs.append(encoder_output)

        encoder_outputs = torch.stack(encoder_outputs)  # 堆叠 encoder 输出
        # print(f"Final encoder outputs shape: {encoder_outputs.shape}")
        return encoder_hidden_state, encoder_cell_state, encoder_outputs

    def decoder(self, encoder_hidden_state, encoder_cell_state, encoder_outputs, labels=None, batches_seen=None):
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim), device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_cell_state = encoder_cell_state

        outputs = []
        attention_weights_list = []
        for t in range(self.decoder_model.horizon):
            decoder_output, (decoder_hidden_state, decoder_cell_state), attention_weights = self.decoder_model(
                go_symbol, encoder_outputs, decoder_hidden_state, decoder_cell_state)
            outputs.append(decoder_output)
            attention_weights_list.append(attention_weights)
            go_symbol = decoder_output

        return torch.stack(outputs), torch.stack(attention_weights_list)

    def forward(self, inputs, labels=None, batches_seen=None):
        encoder_hidden_state, encoder_cell_state, encoder_outputs = self.encoder(inputs)
        outputs, attention_weights = self.decoder(encoder_hidden_state, encoder_cell_state, encoder_outputs, labels, batches_seen=batches_seen)

        # print(f"Final outputs shape: {outputs.shape}")
        # print(f"Attention weights shape: {attention_weights.shape}")

        return outputs, attention_weights  # 返回 outputs 和 注意力权重
