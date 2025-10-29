# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class FreMLP(nn.Module):
    """FreMLP模块：频域MLP操作，用于处理频率域特征

    Args:
        embed_size: 嵌入维度
        sparsity_threshold: 稀疏性阈值
    """

    def __init__(self, embed_size, sparsity_threshold=0.01):
        super(FreMLP, self).__init__()
        self.embed_size = embed_size
        self.sparsity_threshold = sparsity_threshold
        self.scale = 0.02

        # 频域MLP参数
        self.r = nn.Parameter(self.scale * torch.randn(embed_size, embed_size))
        self.i = nn.Parameter(self.scale * torch.randn(embed_size, embed_size))
        self.rb = nn.Parameter(self.scale * torch.randn(embed_size))
        self.ib = nn.Parameter(self.scale * torch.randn(embed_size))

    def forward(self, x):
        # x: [B, nd, dimension // 2 + 1, embed_size]
        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, self.r) - \
            torch.einsum('bijd,dd->bijd', x.imag, self.i) + \
            self.rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, self.r) + \
            torch.einsum('bijd,dd->bijd', x.real, self.i) + \
            self.ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y


class FreTSBlock(nn.Module):
    """FreTSBlock模块：使用频域MLP处理时间序列特征

    Args:
        configs: 模型配置参数
    """

    def __init__(self, configs):
        super(FreTSBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.feature_size = configs.enc_in
        self.embed_size = configs.d_model if hasattr(configs, 'd_model') else 128
        self.sparsity_threshold = 0.01
        self.channel_independence = getattr(configs, 'channel_independence', '1')

        # 频域MLP模块
        self.mlp_temporal = FreMLP(self.embed_size, self.sparsity_threshold)
        if self.channel_independence == '0':
            self.mlp_channel = FreMLP(self.embed_size, self.sparsity_threshold)

        # 嵌入参数
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        x = self.mlp_temporal(x)
        x = torch.fft.irfft(x, n=self.seq_len, dim=2, norm="ortho")
        return x

    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on N dimension
        x = self.mlp_channel(x)
        x = torch.fft.irfft(x, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x)
        bias = x
        # [B, N, T, D]
        if self.channel_independence == '0':
            x = self.MLP_channel(x, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        return x


class FreTS(nn.Module):
    """FreTS模型：专门用于时间序列分类任务的模型
    Args:
        configs: 模型配置参数，包含seq_len, enc_in, d_model, num_class等
    """

    def __init__(self, configs):
        super(FreTS, self).__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model if hasattr(configs, 'd_model') else 128
        self.num_class = configs.num_class
        self.channel_independence = getattr(configs, 'channel_independence', '1')

        # FreTSBlock模块
        self.model = FreTSBlock(configs)

        # 分类任务专用投影层
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.hidden_size = 256

        # 先通过一个线性层将序列特征压缩
        self.feature_compress = nn.Linear(self.seq_len * self.d_model, self.hidden_size)
        # 最终分类投影层
        self.projection = nn.Linear(self.enc_in * self.hidden_size, self.num_class)

    def forward(self, x_enc, x_mark_enc=None):
        """前向传播函数：处理输入序列并输出分类结果

        Args:
            x_enc: 输入序列数据，形状为 [B, T, C]
            x_mark_enc: 时间标记数据，用于掩码填充部分，形状为 [B, T]

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        # 通过FreTSBlock处理
        enc_out = self.model(x_enc)  # [B, N, T, D]

        # 重塑并压缩特征
        B, N, T, D = enc_out.shape
        enc_out = enc_out.reshape(B, N, -1)  # [B, N, T*D]
        enc_out = self.feature_compress(enc_out)  # [B, N, hidden_size]
        enc_out = self.act(enc_out)

        # 分类头处理
        output = self.dropout(enc_out)
        # 应用掩码（如果提供）
        if x_mark_enc is not None:
            output = output * x_mark_enc.unsqueeze(-1)
        # 展平特征并投影到类别空间
        output = output.reshape(output.shape[0], -1)  # [B, N * hidden_size]
        output = self.projection(output)  # [B, num_class]
        return output
