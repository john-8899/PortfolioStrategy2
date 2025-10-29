# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from einops import rearrange


class PAttn(nn.Module):
    """
    PAttn模型：基于patch注意力机制的时间序列分类模型

    原论文链接: https://arxiv.org/abs/2406.16964

    本模型是PAttn的分类任务专用版本，专注于时间序列分类。
    通过将时间序列分割为多个patch，并使用注意力机制捕获patch间的依赖关系，
    从而实现对时间序列的高效分类。

    Args:
        configs: 模型配置参数，包含seq_len, d_model, num_class等
        patch_len: patch的长度，默认为16
        stride: patch的滑动步长，默认为8
    """

    def __init__(self, configs):
        super().__init__()
        self.enc_in = configs.enc_in  # 输入维度
        self.seq_len = configs.seq_len
        self.num_class = configs.num_class
        self.patch_size: int = configs.patch_len
        self.stride: int = configs.stride

        self.d_model: int = configs.d_model

        # 计算patch数量，考虑padding
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 2
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))

        # 输入层：将patch映射到d_model维度
        self.in_layer = nn.Linear(self.patch_size, self.d_model)

        # 编码器：使用注意力机制处理patch序列
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # 分类任务专用组件
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)

        # 输出层：将特征投影到类别空间
        self.out_layer = nn.Linear(self.enc_in*self.d_model * self.patch_num, configs.num_class)

    def forward(self, x_enc, x_mark_enc=None):
        """前向传播函数：处理输入序列并输出分类结果

        Args:
            x_enc: 输入序列数据，形状为 [B, T, C]
            x_mark_enc: 时间标记数据，用于掩码填充部分，形状为 [B, T] (可选)

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        # 标准化处理
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        B, _, C = x_enc.shape

        # 转换维度以便进行patch处理
        x_enc = x_enc.permute(0, 2, 1)  # [B, C, T]

        # 填充以确保所有patch长度一致
        x_enc = self.padding_patch_layer(x_enc)

        # 将时间序列分割为多个patch
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # x_enc形状: [B, C, patch_num, patch_size]

        # 将每个patch映射到d_model维度
        enc_out = self.in_layer(x_enc)
        # enc_out形状: [B, C, patch_num, d_model]

        # 重排以便输入到Transformer编码器
        enc_out = rearrange(enc_out, 'b c m l -> (b c) m l')
        # enc_out形状: [B*C, patch_num, d_model]

        # 通过Transformer编码器
        dec_out, _ = self.encoder(enc_out)
        # dec_out形状: [B*C, patch_num, d_model]

        # 重排回原始批次结构
        dec_out = rearrange(dec_out, '(b c) m l -> b c (m l)', b=B, c=C)
        # dec_out形状: [B, C, patch_num*d_model]

        # 应用激活函数和dropout
        dec_out = self.act(dec_out)
        dec_out = self.dropout(dec_out)

        # 应用掩码（如果提供）
        if x_mark_enc is not None:
            # 将掩码扩展到特征维度
            mask = x_mark_enc.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1]
            # 将掩码调整为与dec_out兼容的形状
            # 计算每个patch对应的掩码元素数量
            mask_elements_per_patch = self.patch_size * self.d_model // self.stride
            mask = mask.repeat(1, C, 1, mask_elements_per_patch)
            mask = mask.view(B, C, -1)  # [B, C, patch_num*d_model]
            dec_out = dec_out * mask

        # 展平特征
        dec_out = dec_out.reshape(dec_out.shape[0], -1)  # [B, C*patch_num*d_model]

        # 投影到类别空间
        output = self.out_layer(dec_out)  # [B, num_class]

        return output