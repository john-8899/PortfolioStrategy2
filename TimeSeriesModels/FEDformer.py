# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock
from layers.MultiWaveletCorrelation import  MultiWaveletTransform
from layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm


class FEDformer(nn.Module):
    """FEDformer模型：专门用于时间序列分类任务的模型

    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html

    Args:
        configs: 模型配置参数，包含seq_len, d_model, num_class等
    """

    def __init__(self, configs, version='Wavelets', mode_select='random', modes=32):
        """初始化FEDformer分类模型

        Args:
            configs: 模型配置参数
            version: FEDformer版本，选项: [Fourier, Wavelets]
            mode_select: 模式选择方法，选项: [random, low]
            modes: 选择的模式数量 ，modes决定了从傅里叶变换结果中选取多少个频率分量进行处理 默认32
        """
        super(FEDformer, self).__init__()
        self.seq_len = configs.seq_len
        self.num_class = configs.num_class
        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Decomp
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        #不同的自注意力机制：
        if self.version == 'Wavelets':
            #如果是'Wavelets'版本：使用小波变换
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
        else:
            #如果是'Fourier'版本(默认)：使用傅里叶变换
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            n_heads=configs.n_heads,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
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
        # enc
        enc_out = self.enc_embedding(x_enc, None)
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