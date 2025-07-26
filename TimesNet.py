# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    """
    使用快速傅里叶变换(FFT)检测输入序列的主要周期，并返回周期和对应的频域特征

    Args:
        x (torch.Tensor): 输入张量，形状为[B, T, C]，其中B是batch大小，T是时间步长，C是特征维度
        k (int, optional): 需要返回的top-k周期数量，默认为2

    Returns:
        tuple: 包含两个元素的元组
            - period (numpy.ndarray): 检测到的top-k周期长度，形状为[k]
            - frequency_features (torch.Tensor): 对应top-k频率的频域特征，形状为[B, k]
    """
    # 计算输入序列的实数快速傅里叶变换
    xf = torch.fft.rfft(x, dim=1)

    # 通过振幅计算频率重要性并找出top-k频率
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0  # 忽略直流分量
    _, top_list = torch.topk(frequency_list, k)

    # 计算实际周期长度并准备返回结果
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list

    return period, abs(xf).mean(-1)[:, top_list]



class TimesBlock(nn.Module):
    """TimesBlock模块：使用FFT提取周期特征并通过2D卷积处理

    Args:
        configs: 模型配置参数，包含d_model, d_ff, num_kernels (in_channels, out_channels, num_kernels)等
    """
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.k = configs.top_k
        # 参数高效设计的卷积块
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # 填充以适应周期长度
            if T % period != 0:
                length = ((T // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - T), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = T
                out = x
            # 重塑为2D结构以应用卷积
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            # 重塑回原始形状
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :T, :])
        res = torch.stack(res, dim=-1)
        # 自适应聚合不同周期的特征
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # 残差连接
        res = res + x
        return res


class CTimesNet(nn.Module):
    """TimesNet2模型：专门用于时间序列分类任务的模型

    移除了原TimesNet中的预测、异常检测等功能，专注于分类任务，
    优化了网络结构以提高分类性能和计算效率。

    Args:
        configs: 模型配置参数，包含seq_len, d_model, num_class等
    """
    def __init__(self, configs):
        super(CTimesNet, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.num_class = configs.num_class
        self.d_model = configs.d_model
        # 创建TimesBlock堆叠
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        # 数据嵌入层
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        # 分类任务专用投影层
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forward(self, x_enc, x_mark_enc=None):
        """前向传播函数：处理输入序列并输出分类结果

        Args:
            x_enc: 输入序列数据，形状为 [B, T, C]
            x_mark_enc: 时间标记数据，用于掩码填充部分，形状为 [B, T]

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        # 数据嵌入
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, C]
        # 通过TimesBlock堆叠
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # 分类头处理
        output = self.act(enc_out)
        output = self.dropout(output)
        # 应用掩码（如果提供）
        if x_mark_enc is not None:
            output = output * x_mark_enc.unsqueeze(-1)
        # 展平特征并投影到类别空间
        output = output.reshape(output.shape[0], -1)  # [B, seq_len * d_model]
        output = self.projection(output)  # [B, num_class]
        return output