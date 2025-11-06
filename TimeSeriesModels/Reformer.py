# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import ReformerLayer
from layers.Embed import DataEmbedding


class Reformer(nn.Module):
    """Reformer模型：专门用于时间序列分类任务的模型
    Args:
        configs: 模型配置参数，包含seq_len, d_model, num_class等
    """

    def __init__(self, configs, bucket_size=4, n_hashes=4):
        """初始化Reformer分类模型

        Args:
            configs: 模型配置参数
            bucket_size: Reformer中局部敏感哈希的桶大小（2-16）
            n_hashes: Reformer中局部敏感哈希的哈希次数（2-8）
            起始设置: 从默认值 bucket_size=4, n_hashes=4 开始
            根据序列长度调整:
            对于较短序列(如 seq_len < 128): 可以使用较小的值
            对于较长序列(如 seq_len > 512): 可能需要增加这两个参数
            性能与精度权衡:
            如果追求速度: 降低两个参数值
            如果追求精度: 适当增加参数值
            bucket_size=4, n_hashes=4 (默认配置)
            bucket_size=8, n_hashes=2 (注重效率)
            bucket_size=2, n_hashes=8 (注重精度)
        """
        super(Reformer, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.num_class = configs.num_class
        self.d_model = configs.d_model

        # 数据嵌入层
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # 编码器：使用Reformer层替换传统注意力机制
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, configs.d_model, configs.n_heads,
                                  bucket_size=bucket_size, n_hashes=n_hashes),
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

        # 通过编码器处理
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