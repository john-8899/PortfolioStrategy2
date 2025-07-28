# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch.nn.functional as F


class Projector(nn.Module):
    """用于学习非平稳时间序列的De-stationary因子的MLP模块

    论文链接: https://openreview.net/pdf?id=ucNDIDRNjjv
    通过卷积和全连接层学习时间序列的统计特性
    Args:
        enc_in (int): 输入序列的特征维度
        seq_len (int): 输入序列的长度
        hidden_dims (list): 隐藏层维度列表
        hidden_layers (int): 隐藏层层数
        output_dim (int): 输出维度
        kernel_size (int, optional): 卷积核大小. Defaults to 3.
    """
    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()
        # 根据PyTorch版本设置卷积填充大小
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # 使用1D卷积提取序列特征
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)
        # 构建MLP骨干网络
        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        """前向传播函数

        Args:
            x: 输入序列数据，形状为 [B, S, E]
            stats: 统计特征数据，形状为 [B, 1, E]

        Returns:
            y: 学习到的De-stationary因子，形状为 [B, O]
        """
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y


class Nonstationary_Transformer(nn.Module):
    """Nonstationary_Transformer模型：专门用于时间序列分类任务

    专注于提高分类任务的性能和效率。
    """
    def __init__(self, configs):
        super(Nonstationary_Transformer, self).__init__()
        self.seq_len = configs.seq_len
        self.num_class = configs.num_class

        # 数据嵌入层
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # 编码器
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # 非平稳因子学习器
        self.tau_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len,
                                     hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                     output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len,
                                      hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                      output_dim=configs.seq_len)

        # 分类头
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
        x_raw = x_enc.clone().detach()

        # 数据归一化
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        std_enc = torch.sqrt(torch.var(x_enc - mean_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = (x_enc - mean_enc) / std_enc

        # 学习非平稳因子
        tau = self.tau_learner(x_raw, std_enc)
        tau = torch.clamp(tau, max=80.0).exp()  # 限制tau值避免数值溢出
        delta = self.delta_learner(x_raw, mean_enc)

        # 数据嵌入与编码器处理
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

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