import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    """基础卷积模块，包含卷积、批归一化和激活函数"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    """通道池化模块，用于空间注意力中的最大池化和平均池化"""
    def forward(self, x):
        return torch.cat((torch.max(x, dim=1)[0].unsqueeze(1), torch.mean(x, dim=1).unsqueeze(1)), dim=1)

class SpatialAttention(nn.Module):
    """空间注意力模块（SA）"""
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size=3, stride=1, padding=1, relu=False)

    def forward(self, x):
        # print(f"[SpatialAttention] Input shape: {x.shape}")
        x_compress = self.compress(x)  # 通道池化 (B, 2, H, W)
        # print(f"[SpatialAttention] After ChannelPool (B, 2, H, W): {x_compress.shape}")
        x_out = self.spatial(x_compress)  # 卷积处理 (B, 1, H, W)
        # print(f"[SpatialAttention] After BasicConv (B, 1, H, W): {x_out.shape}")
        scale = torch.sigmoid(x_out)  # 激活生成空间注意力图
        # print(f"[SpatialAttention] Scale shape: {scale.shape}")
        return scale  # 输出空间注意力权重


class TemporalAttention(nn.Module):
    """时间注意力模块，输出 (B, T, 1, 1)"""
    def __init__(self, gate_channels, reduction_ratio=1):
        super(TemporalAttention, self).__init__()
        self.gate_channels = gate_channels
        self.shared_conv = nn.Conv2d(1, 1, kernel_size=1)  # 使用共享的 1x1 卷积层

    def forward(self, x):
        # print(f"[TemporalAttention] Input x shape (B, T, N, F): {x.shape}")

        # 首先在特征维度 F 上求均值，将维度减少到 (B, T, N, 1)
        x = torch.mean(x, dim=3, keepdim=True)
        # print(f"[TemporalAttention] After mean over F, shape: {x.shape}")

        # 在节点维度 N 上进行全局池化，得到 (B, T, 1, 1)
        avg_pool = torch.mean(x, dim=2, keepdim=True)  # (B, T, 1, 1)
        max_pool = torch.max(x, dim=2, keepdim=True)[0]  # (B, T, 1, 1)
        # print(f"[TemporalAttention] avg_pool shape: {avg_pool.shape}")
        # print(f"[TemporalAttention] max_pool shape: {max_pool.shape}")

        # 将 (B, T, 1, 1) 变换为 (B*T, 1, 1, 1)，使得 T 维度被当作 batch 处理
        avg_pool = avg_pool.view(-1, 1, 1, 1)
        max_pool = max_pool.view(-1, 1, 1, 1)

        # 通过共享的 1x1 卷积层
        avg_out = self.shared_conv(avg_pool)  # (B*T, 1, 1, 1)
        max_out = self.shared_conv(max_pool)  # (B*T, 1, 1, 1)
        # print(f"[TemporalAttention] avg_out shape after shared_conv: {avg_out.shape}")
        # print(f"[TemporalAttention] max_out shape after shared_conv: {max_out.shape}")

        # 重新 reshape 回 (B, T, 1, 1)
        avg_out = avg_out.view(-1, x.size(1), 1, 1)  # (B, T, 1, 1)
        max_out = max_out.view(-1, x.size(1), 1, 1)  # (B, T, 1, 1)

        # 元素求和并使用 sigmoid 激活生成时间注意力权重
        channel_att = avg_out + max_out
        scale = torch.sigmoid(channel_att)  # (B, T, 1, 1)
        return scale



class SpatialTemporalAttention(nn.Module):
    """时空注意力模块（图5实现）"""
    def __init__(self, gate_channels, reduction_ratio=2):
        super(SpatialTemporalAttention, self).__init__()
        self.spatial_att = SpatialAttention()  # 空间注意力模块
        self.temporal_att = TemporalAttention(gate_channels, reduction_ratio)  # 时间注意力模块

    def forward(self, x):

        # 转置到 (B, T, N, F) 以符合 TemporalAttention 的输入需求
        x = x.permute(1, 0, 2, 3)
        # print(f"[SpatialTemporalAttention] Input shape: {x.shape}")

        # 1. 空间注意力
        spatial_weight = self.spatial_att(x)  # (B, 1, H, W)
        # print(f"[SpatialTemporalAttention] Spatial weight shape: {spatial_weight.shape}")

        # 2. 时间注意力
        temporal_weight = self.temporal_att(x)  # (B, F, 1, 1)
        # print(f"[SpatialTemporalAttention] Temporal weight shape: {temporal_weight.shape}")

        # 3. 时空融合
        st_weight = torch.sigmoid(spatial_weight * temporal_weight)  # (B, F, H, W)
        # print(f"[SpatialTemporalAttention] ST weight shape: {st_weight.shape}")

        # 4. 加权输入特征
        refined_x = x * st_weight  # (B, F, H, W)(T, B, N, F)
        # print(f"[SpatialTemporalAttention] Refined x shape: {refined_x.shape}")
        # print("11111")
        # 5. 将维度转换为目标格式 (T, B, N, F)
        refined_x = refined_x.permute(1,0,2,3)
        # print(f"[SpatialTemporalAttention] Final shape (T, B, N, F): {refined_x.shape}")

        return refined_x
