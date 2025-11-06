# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from einops import rearrange, repeat
from layers.Crossformer_EncDec import scale_block, Encoder
from layers.Embed import PatchEmbedding
from math import ceil


class Crossformer(nn.Module):
    """
    Crossformer模型：专门用于时间序列分类任务的模型

    基于Crossformer架构，专注于分类任务。
    使用两阶段注意力机制和分层结构，有效捕获时间序列的长期依赖关系。

    Args:
        configs: 模型配置参数，包含以下关键参数：
            - enc_in: 输入特征维度
            - seq_len: 输入序列长度
            - seg_len: 分段长度，默认为12 （4-24）
                作用: 将输入时间序列分割成固定长度的片段，用于处理时间序列数据的分块处理
                默认值: 等于序列长度 seq_len
                处理方式: 代码中通过 PatchEmbedding 对输入序列按 seg_len 进行分段嵌入
            - win_size: 窗口大小，默认为2 （2-8）
                作用: 控制注意力机制中的窗口大小，影响模型捕获时间依赖关系的范围
                默认值: 2
                处理方式: 在 Encoder 的 scale_block 中使用，控制分层注意力机制的窗口大小
            - d_model: 模型隐藏维度
            - n_heads: 注意力头数
            - e_layers: 编码器层数
            - d_ff: 前馈网络维度
            - dropout: Dropout概率
            - factor: 注意力因子
            - num_class: 分类类别数
    """

    def __init__(self, configs):
        super(Crossformer, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.seg_len: int = configs.seq_len  # 分段长度
        self.win_size: int = configs.win_size  # 窗口大小
        self.d_model = configs.d_model
        self.num_class = configs.num_class

        # 处理不可见序列长度的填充操作
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.head_nf = configs.d_model * self.in_seg_num

        # 嵌入层
        self.enc_value_embedding = PatchEmbedding(
            configs.d_model, self.seg_len, self.seg_len,
            self.pad_in_len - configs.seq_len, 0
        )
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model)
        )
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # 编码器：使用两阶段注意力机制的分层结构
        self.encoder = Encoder([
            scale_block(
                configs,
                1 if l == 0 else self.win_size,
                configs.d_model,
                configs.n_heads,
                configs.d_ff,
                1,
                configs.dropout,
                self.in_seg_num if l == 0 else ceil(self.in_seg_num / self.win_size ** l),
                configs.factor
            ) for l in range(configs.e_layers)
        ])

        # 分类头
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

    def forward(self, x_enc, x_mark_enc=None):
        """
        前向传播函数：处理输入序列并输出分类结果

        Args:
            x_enc: 输入序列数据，形状为 [B, T, C] 或 [B, C, T]
            x_mark_enc: 时间标记数据，用于掩码填充部分，形状为 [B, T] (可选)

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        # 嵌入处理
        # 如果输入是 [B, T, C] 格式，转换为 [B, C, T]
        if x_enc.shape[1] == self.seq_len:
            x_enc = x_enc.permute(0, 2, 1)

        # 值嵌入和位置嵌入
        x_enc, n_vars = self.enc_value_embedding(x_enc)
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)

        # 编码器处理
        enc_out, _ = self.encoder(x_enc)

        # 分类头处理
        output = self.flatten(enc_out[-1].permute(0, 1, 3, 2))
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)

        return output