# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy import special as ss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def transition(N):
    Q = np.arange(N, dtype=np.float64)
    R = (2 * Q + 1)[:, None]  # / theta
    j, i = np.meshgrid(Q, Q)
    A = np.where(i < j, -1, (-1.) ** (i - j + 1)) * R
    B = (-1.) ** Q[:, None] * R
    return A, B


class HiPPO_LegT(nn.Module):
    def __init__(self, N, dt=1.0, discretization='bilinear'):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super(HiPPO_LegT, self).__init__()
        self.N = N
        A, B = transition(N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        B = B.squeeze(-1)

        self.register_buffer('A', torch.Tensor(A).to(device))
        self.register_buffer('B', torch.Tensor(B).to(device))
        vals = np.arange(0.0, 1.0, dt)
        self.register_buffer('eval_matrix', torch.Tensor(
            ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T).to(device))

    def forward(self, inputs):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """
        c = torch.zeros(inputs.shape[:-1] + tuple([self.N])).to(device)
        cs = []
        for f in inputs.permute([-1, 0, 1]):
            f = f.unsqueeze(-1)
            new = f @ self.B.unsqueeze(0)
            c = F.linear(c, self.A) + new
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, ratio=0.5):
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.modes = min(32, seq_len // 2)
        self.index = list(range(0, self.modes))

        self.scale = (1 / (in_channels * out_channels))
        self.weights_real = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))
        self.weights_imag = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))

    def compl_mul1d(self, order, x, weights_real, weights_imag):
        return torch.complex(torch.einsum(order, x.real, weights_real) - torch.einsum(order, x.imag, weights_imag),
                             torch.einsum(order, x.real, weights_imag) + torch.einsum(order, x.imag, weights_real))

    def forward(self, x):
        B, H, E, N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, H, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        a = x_ft[:, :, :, :self.modes]
        out_ft[:, :, :, :self.modes] = self.compl_mul1d("bjix,iox->bjox", a, self.weights_real, self.weights_imag)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FiLM(nn.Module):
    """
    FiLM模型：专门用于时间序列分类任务的模型
    Paper link: https://arxiv.org/abs/2205.08897
    Args:
        configs: 模型配置参数，包含seq_len, enc_in, num_class等
    """

    def __init__(self, configs):
        super(FiLM, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.num_class = configs.num_class

        # 仿射变换参数，用于输入数据的归一化和反归一化
        # affine_weight: 仿射权重参数，形状为 [1, 1, enc_in]
        # affine_bias: 仿射偏置参数，形状为 [1, 1, enc_in]
        # b, s, f means b, f
        self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))

        # 多尺度和窗口大小配置
        # multiscale: 多尺度因子列表，用于扩展输入序列长度
        # window_size: 窗口大小列表，用于HiPPO-LegT变换
        self.multiscale = configs.multiscale
        self.window_size = configs.window_size
        configs.ratio = configs.ratio

        # 构建HiPPO-LegT变换层列表，用于不同尺度和窗口大小的组合
        self.legts = nn.ModuleList(
            [HiPPO_LegT(N=n, dt=1. / self.seq_len / i) for n in self.window_size for i in self.multiscale])
        # 构建谱卷积层列表，与HiPPO-LegT层一一对应
        self.spec_conv_1 = nn.ModuleList([SpectralConv1d(in_channels=n, out_channels=n,
                                                         seq_len=self.seq_len,
                                                         ratio=configs.ratio) for n in
                                          self.window_size for _ in range(len(self.multiscale))])
        # 多层感知机，用于融合不同尺度的输出
        self.mlp = nn.Linear(len(self.multiscale) * len(self.window_size), 1)

        # 分类任务专用投影层
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(
            configs.enc_in * configs.seq_len, configs.num_class)

    def forward(self, x_enc, x_mark_enc=None):
        """前向传播函数：处理输入序列并输出分类结果

        Args:
            x_enc: 输入序列数据，形状为 [B, T, C]
            x_mark_enc: 时间标记数据，用于掩码填充部分，形状为 [B, T]

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev
        # 应用仿射变换
        x_enc = x_enc * self.affine_weight + self.affine_bias
        x_decs = []
        jump_dist = 0
        for i in range(0, len(self.multiscale) * len(self.window_size)):
            # 根据多尺度因子计算输入长度
            x_in_len = self.multiscale[i % len(self.multiscale)] * self.seq_len
            # 截取输入序列的最后x_in_len个时间步
            x_in = x_enc[:, -x_in_len:]
            legt = self.legts[i]
            # 应用HiPPO-LegT变换
            x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2, 3, 0])[:, :, :, jump_dist:]
            # 应用谱卷积
            out1 = self.spec_conv_1[i](x_in_c)
            # 提取卷积输出的特定时间步
            if self.seq_len >= self.seq_len:
                x_dec_c = out1.transpose(2, 3)[:, :, self.seq_len - 1 - jump_dist, :]
            else:
                x_dec_c = out1.transpose(2, 3)[:, :, -1, :]
            # 应用评估矩阵进行解码
            x_dec = x_dec_c @ legt.eval_matrix[-self.seq_len:, :].T
            x_decs.append(x_dec)
        # 融合不同尺度的输出结果
        x_dec = torch.stack(x_decs, dim=-1)
        x_dec = self.mlp(x_dec).squeeze(-1).permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        x_dec = x_dec - self.affine_bias
        x_dec = x_dec / (self.affine_weight + 1e-10)
        x_dec = x_dec * stdev
        x_dec = x_dec + means

        # 分类头处理
        output = self.act(x_dec)
        output = self.dropout(output)
        # 应用掩码（如果提供）
        if x_mark_enc is not None:
            output = output * x_mark_enc.unsqueeze(-1)
        # 展平特征并投影到类别空间
        output = output.reshape(output.shape[0], -1)  # [B, seq_len * enc_in]
        output = self.projection(output)  # [B, num_class]
        return output