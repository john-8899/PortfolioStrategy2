import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """
    """
    参数:
    - input_dim: 输入维度
    - seq_len: 序列长度
    - d_model: 模型的嵌入维度，默认为128 (32-128) (conv1d:in_channels)
    - dropout: Dropout比例，默认为0.1
    - moving_avg: 移动平均的窗口大小，默认为5(要小于seq_len)(相当于：kernel_size)
    - factor: 一个缩放因子，默认为1.0
            factor 用于缩放输入序列的时间维度。通过调整 factor，可以控制傅里叶变换时考虑的频率范围。较大的 factor 会关注较低频率的周期性模式，而较小的 factor 则会更敏感于高频的周期性变化。
            高 factor 设置：可能更适合捕捉日周期（低频）的变化，忽略一些高频噪声。
            低 factor 设置：可能更适合捕捉周周期（相对高频）的变化，同时保留更多的细节信息。
    - output_attention: 是否输出注意力权重，默认为False
    - n_heads: 注意力头的数量，默认为8  ()
    - d_ff: Feedforward层的维度，默认为256 ( Conv1d:out_channels)
    - activation: Feedforward层使用的激活函数，默认为'gelu'
    - e_layers: Encoder层的数量，默认为3  (2-3)
    - num_class: 分类的数量，默认为4
    """
    def __init__(self,input_dim,seq_len,d_model=128,dropout=0.1,moving_avg=5,factor=1.0,output_attention=False,n_heads= 8,d_ff=256,activation='gelu',e_layers=3,num_class=3):
        super(Autoformer, self).__init__()

        self.seq_len = seq_len
        self.output_attention = output_attention

        ##Decomp
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)# 修改过源代码

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(input_dim,d_model, embed_type='fixed', freq='h', dropout=dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout = dropout,
                                        output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model * seq_len, num_class)
    def classification(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        #output = output * x_mark_enc.unsqueeze(-1)
        output = output.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        #print(f"Shape of output before projection: {output.shape}")
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc):

        dec_out = self.classification(x_enc)
        return dec_out  # [B, N]
