# -*- coding: utf-8 -*-
import numpy as np
# import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.MSGBlock import GraphBlock, simpleVIT, Attention_Block, Predict


def FFT_for_Period(x, k=2):
    """使用快速傅里叶变换(FFT)提取时间序列中的主要周期

    通过分析频域特征识别时间序列中的周期性模式，
    为多尺度图卷积提供尺度信息。

    Args:
        x: 输入时间序列，形状为 [B, T, C]
        k: 返回的top-k个主要周期

    Returns:
        period: 主要周期列表
        period_weight: 周期权重，用于自适应聚合
    """
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # 通过幅值找到周期
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0  # 去除直流分量
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class ScaleGraphBlock(nn.Module):
    """多尺度图卷积块：融合图卷积和注意力机制的特征提取模块

    该模块通过FFT提取多尺度周期信息，在不同尺度上应用图卷积捕获序列的
    空间关联性，同时结合注意力机制增强特征表示能力。

    Args:
        configs: 模型配置参数
            - seq_len: 序列长度
            - k: 尺度数量
                控制从FFT中提取的主要周期数量，决定多尺度图卷积的层数
                通常设置为2-5，根据数据的周期性复杂度调整
            - d_model: 模型维度
                模型的隐藏层维度，影响特征表示能力
                合理范围: 通常为32, 64, 128, 256等2的幂次，根据数据复杂度和计算资源选择
            - d_ff: 前馈网络维度
                作用: 注意力机制中前馈网络的中间维度
                合理范围: 通常是d_model的2-4倍，如128, 256, 512
            - c_out: 输出通道数
                作用: 图卷积层的输出通道数
                合理范围: 与d_model相近或相同，如32, 64, 128
            - conv_channel: 卷积通道数
                作用: GraphBlock中卷积层的通道数
                合理范围: 通常为32-128，取决于模型复杂度需求
            - skip_channel: 跳跃连接通道数
                作用: 跳跃连接中的通道数，用于信息传递
                合理范围: 与conv_channel相近，通常为32-128
            - gcn_depth: 图卷积深度
                作用: 图卷积网络的层数
                合理范围: 通常为2-10层，过深可能导致过拟合
            - dropout: Dropout比例
            - propalpha: 传播衰减因子
                作用: 图卷积中信息传播的衰减系数
                合理范围: 0-1之间，通常为0.05或0.1
            - node_dim: 节点维度
                作用: 图结构中节点的嵌入维度
                合理范围: 通常为10-100，取决于节点特征复杂度
    """

    def __init__(self, configs):
        super(ScaleGraphBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.k = configs.top_k

        # 多头注意力机制用于特征增强
        self.att0 = Attention_Block(configs.d_model, configs.d_ff,
                                    n_heads=configs.n_heads, dropout=configs.dropout, activation="gelu")
        self.norm = nn.LayerNorm(configs.d_model)
        self.gelu = nn.GELU()

        # 多尺度图卷积层
        self.gconv = nn.ModuleList()
        for i in range(self.k):
            self.gconv.append(
                GraphBlock(configs.c_out, configs.d_model, configs.conv_channel, configs.skip_channel,
                           configs.gcn_depth, configs.dropout, configs.propalpha, configs.seq_len,
                           configs.node_dim))

    def forward(self, x):
        """前向传播：多尺度图卷积与注意力融合

        Args:
            x: 输入特征，形状为 [B, T, N]

        Returns:
            输出特征，形状为 [B, T, N]
        """
        B, T, N = x.size()
        scale_list, scale_weight = FFT_for_Period(x, self.k)
        res = []

        for i in range(self.k):
            scale = scale_list[i]
            # 图卷积处理
            x_gconv = self.gconv[i](x)

            # 填充以适应尺度长度
            if self.seq_len % scale != 0:
                length = ((self.seq_len // scale) + 1) * scale
                padding = torch.zeros([x.shape[0], (length - self.seq_len), x.shape[2]]).to(x.device)
                out = torch.cat([x_gconv, padding], dim=1)
            else:
                length = self.seq_len
                out = x_gconv

            # 重塑为2D结构
            out = out.reshape(B, length // scale, scale, N)

            # 多头注意力处理
            out = out.reshape(-1, scale, N)
            out = self.norm(self.att0(out))
            out = self.gelu(out)
            out = out.reshape(B, -1, scale, N).reshape(B, -1, N)

            # 截取原始序列长度
            out = out[:, :self.seq_len, :]
            res.append(out)

        # 自适应聚合不同尺度的特征
        res = torch.stack(res, dim=-1)
        scale_weight = F.softmax(scale_weight, dim=1)
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * scale_weight, -1)

        # 残差连接
        res = res + x
        return res


class MSGNet(nn.Module):
    """MSGNet模型：专门用于时间序列分类的多尺度图神经网络

    基于MSGNet架构优化，专门针对时间序列分类任务设计。
    融合了多尺度图卷积、注意力机制和频域分析，提供了强大的
    序列特征提取和分类能力。

    架构特点：
    - 多尺度图卷积：捕获不同时间尺度的模式
    - 频域分析：自动识别周期性特征
    - 注意力机制：增强关键特征表示
    - 图结构建模：建模变量间的空间关联

    Args:
        configs: 模型配置参数
            - seq_len: 输入序列长度
            - d_model: 模型隐藏维度
            - e_layers: 编码器层数
            - num_class: 分类类别数
            - enc_in: 输入特征数
            - embed: 嵌入维度
            - freq: 时间特征编码频率
            - dropout: Dropout比例
    """

    def __init__(self, configs):
        super(MSGNet, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.num_class = configs.num_class
        self.d_model = configs.d_model

        # 多尺度图卷积块堆叠
        self.model = nn.ModuleList([ScaleGraphBlock(configs) for _ in range(configs.e_layers)])

        # 数据嵌入层
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model,
                                           configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # 分类任务专用层
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forward(self, x_enc, x_mark_enc=None):
        """前向传播：时间序列分类

        Args:
            x_enc: 输入时间序列，形状为 [B, T, C]
            x_mark_enc: 时间标记，用于掩码填充部分，形状为 [B, T]

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # 数据嵌入
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, T, C]

        # 通过多尺度图卷积块
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