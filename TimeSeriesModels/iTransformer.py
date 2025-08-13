# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class iTransformer(nn.Module):
    """iTransformer模型：专门用于时间序列分类任务的模型
    优化了网络结构以提高分类性能和计算效率。
    Paper link: https://arxiv.org/abs/2310.06625
    Args:
        configs: 模型配置参数，包含seq_len, d_model, num_class等
    """

    def __init__(self, configs):
        super(iTransformer, self).__init__()
        self.seq_len = configs.seq_len
        self.num_class = configs.num_class
        self.d_model = configs.d_model

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
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
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # 分类任务专用投影层
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

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
        # 通过编码器
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

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