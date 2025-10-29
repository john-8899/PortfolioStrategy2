# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:10:01 2025
@author: Murad
SISLab, USF
mmurad@usf.edu
https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer

Modified for classification task
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from layers.DWT_Decomposition import Decomposition


class TokenMixer(nn.Module):
    """Token混合器：用于混合序列中的token信息

    Args:
        input_seq: 输入序列长度
        batch_size: 批处理大小
        channel: 通道数
        pred_seq: 预测序列长度
        dropout: Dropout比率
        factor: 扩展因子
        d_model: 模型维度
    """

    def __init__(self, input_seq, batch_size, channel, pred_seq, dropout, factor, d_model):
        super(TokenMixer, self).__init__()
        self.input_seq: int = input_seq
        self.batch_size = batch_size
        self.channel = channel
        self.pred_seq: int = pred_seq
        self.dropout = dropout
        self.factor = factor
        self.d_model = d_model

        self.dropoutLayer = nn.Dropout(self.dropout)
        self.layers = nn.Sequential(nn.Linear(self.input_seq, self.pred_seq * self.factor),
                                    nn.GELU(),
                                    nn.Dropout(self.dropout),
                                    nn.Linear(self.pred_seq * self.factor, self.pred_seq)
                                    )

    def forward(self, x):
        """前向传播

        Args:
            x: 输入张量，形状为 [B, C, T, d]

        Returns:
            Tensor: 输出张量，形状为 [B, C, T, d]
        """
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = x.transpose(1, 2)
        return x


class Mixer(nn.Module):
    """混合器：结合Token混合器和嵌入混合器

    Args:
        input_seq: 输入序列长度
        out_seq: 输出序列长度
        batch_size: 批处理大小
        channel: 通道数
        d_model: 模型维度
        dropout: Dropout比率
        tfactor: Token混合器的扩展因子
        dfactor: 嵌入混合器的扩展因子
    """

    def __init__(self,
                 input_seq,
                 out_seq,
                 batch_size,
                 channel,
                 d_model,
                 dropout,
                 tfactor,
                 dfactor):
        super(Mixer, self).__init__()
        self.input_seq = input_seq
        self.pred_seq = out_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.tfactor = tfactor  # expansion factor for patch mixer
        self.dfactor = dfactor  # expansion factor for embedding mixer

        self.tMixer = TokenMixer(input_seq=self.input_seq, batch_size=self.batch_size, channel=self.channel,
                                 pred_seq=self.pred_seq, dropout=self.dropout, factor=self.tfactor,
                                 d_model=self.d_model)
        self.dropoutLayer = nn.Dropout(self.dropout)
        self.norm1 = nn.BatchNorm2d(self.channel)
        self.norm2 = nn.BatchNorm2d(self.channel)

        self.embeddingMixer = nn.Sequential(nn.Linear(self.d_model, self.d_model * self.dfactor),
                                            nn.GELU(),
                                            nn.Dropout(self.dropout),
                                            nn.Linear(self.d_model * self.dfactor, self.d_model))

    def forward(self, x):
        """前向传播

        Args:
            x: 输入张量，形状为 [B, C, Patch_number, d_model]

        Returns:
            Tensor: 输出张量，形状为 [B, C, Patch_number, d_model]
        """
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dropoutLayer(self.tMixer(x))
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x + self.dropoutLayer(self.embeddingMixer(x))
        return x


class ResolutionBranch(nn.Module):
    """分辨率分支：处理不同分辨率的小波系数

    Args:
        input_seq: 输入序列长度
        pred_seq: 预测序列长度
        batch_size: 批处理大小
        channel: 通道数
        d_model: 模型维度
        dropout: Dropout比率
        embedding_dropout: 嵌入层Dropout比率
        tfactor: Token混合器扩展因子
        dfactor: 嵌入混合器扩展因子
        patch_len: 补丁长度
        patch_stride: 补丁步长
    """

    def __init__(self,
                 input_seq,
                 pred_seq,
                 batch_size,
                 channel,
                 d_model,
                 dropout,
                 embedding_dropout,
                 tfactor,
                 dfactor,
                 patch_len,
                 patch_stride):
        super(ResolutionBranch, self).__init__()
        self.input_seq = input_seq
        self.pred_seq = pred_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_num = int((self.input_seq - self.patch_len) / self.patch_stride + 2)

        self.patch_norm = nn.BatchNorm2d(self.channel)
        self.patch_embedding_layer = nn.Linear(self.patch_len, self.d_model)  # shared among all channels
        self.mixer1 = Mixer(input_seq=self.patch_num,
                            out_seq=self.patch_num,
                            batch_size=self.batch_size,
                            channel=self.channel,
                            d_model=self.d_model,
                            dropout=self.dropout,
                            tfactor=self.tfactor,
                            dfactor=self.dfactor)
        self.mixer2 = Mixer(input_seq=self.patch_num,
                            out_seq=self.patch_num,
                            batch_size=self.batch_size,
                            channel=self.channel,
                            d_model=self.d_model,
                            dropout=self.dropout,
                            tfactor=self.tfactor,
                            dfactor=self.dfactor)
        self.norm = nn.BatchNorm2d(self.channel)
        self.dropoutLayer = nn.Dropout(self.embedding_dropout)
        self.head = nn.Sequential(nn.Flatten(start_dim=-2, end_dim=-1),
                                  nn.Linear(self.patch_num * self.d_model, self.pred_seq))

    def forward(self, x):
        """前向传播

        Args:
            x: 输入系数序列，形状为 [B, channel, length_of_coefficient_series]

        Returns:
            Tensor: 预测系数序列，形状为 [B, channel, length_of_pred_coeff_series]
        """
        x_patch = self.do_patching(x)
        x_patch = self.patch_norm(x_patch)
        x_emb = self.dropoutLayer(self.patch_embedding_layer(x_patch))

        out = self.mixer1(x_emb)
        res = out
        out = res + self.mixer2(out)
        out = self.norm(out)

        out = self.head(out)
        return out

    def do_patching(self, x):
        """进行补丁分割

        Args:
            x: 输入张量

        Returns:
            Tensor: 补丁分割后的张量
        """
        x_end = x[:, :, -1:]
        x_padding = x_end.repeat(1, 1, self.patch_stride)
        x_new = torch.cat((x, x_padding), dim=-1)
        x_patch = x_new.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return x_patch


class WPMixerCore(nn.Module):
    """WPMixer核心模块：处理小波分解和混合

    Args:
        input_length: 输入长度
        pred_length: 预测长度
        wavelet_name: 小波名称
        level: 小波分解层级
        batch_size: 批处理大小
        channel: 通道数
        d_model: 模型维度
        dropout: Dropout比率
        embedding_dropout: 嵌入层Dropout比率
        tfactor: Token混合器扩展因子
        dfactor: 嵌入混合器扩展因子
        device: 设备
        patch_len: 补丁长度
        patch_stride: 补丁步长
        no_decomposition: 是否不进行分解
        use_amp: 是否使用自动混合精度
    """

    def __init__(self,
                 input_length,
                 pred_length,
                 wavelet_name,
                 level,
                 batch_size,
                 channel,
                 d_model,
                 dropout,
                 embedding_dropout,
                 tfactor,
                 dfactor,
                 device,
                 patch_len,
                 patch_stride,
                 no_decomposition,
                 use_amp):
        super(WPMixerCore, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.device = device
        self.no_decomposition = no_decomposition
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.use_amp = use_amp

        self.Decomposition_model = Decomposition(input_length=self.input_length,
                                                 pred_length=self.pred_length,
                                                 wavelet_name=self.wavelet_name,
                                                 level=self.level,
                                                 batch_size=self.batch_size,
                                                 channel=self.channel,
                                                 d_model=self.d_model,
                                                 tfactor=self.tfactor,
                                                 dfactor=self.dfactor,
                                                 device=self.device,
                                                 no_decomposition=self.no_decomposition,
                                                 use_amp=self.use_amp)

        self.input_w_dim = self.Decomposition_model.input_w_dim  # list of the length of the input coefficient series
        self.pred_w_dim = self.Decomposition_model.pred_w_dim  # list of the length of the predicted coefficient series

        self.patch_len = patch_len
        self.patch_stride = patch_stride

        # (m+1) number of resolutionBranch
        self.resolutionBranch = nn.ModuleList([ResolutionBranch(input_seq=self.input_w_dim[i],
                                                                pred_seq=self.pred_w_dim[i],
                                                                batch_size=self.batch_size,
                                                                channel=self.channel,
                                                                d_model=self.d_model,
                                                                dropout=self.dropout,
                                                                embedding_dropout=self.embedding_dropout,
                                                                tfactor=self.tfactor,
                                                                dfactor=self.dfactor,
                                                                patch_len=self.patch_len,
                                                                patch_stride=self.patch_stride) for i in
                                               range(len(self.input_w_dim))])

    def forward(self, xL):
        """前向传播

        Args:
            xL: 查看窗口，形状为 [B, look_back_length, channel]

        Returns:
            Tensor: 预测时间序列，形状为 [B, prediction_length, output_channel]
        """
        x = xL.transpose(1, 2)  # [batch, channel, look_back_length]

        # xA: approximation coefficient series,
        # xD: detail coefficient series
        # yA: predicted approximation coefficient series
        # yD: predicted detail coefficient series

        xA, xD = self.Decomposition_model.transform(x)

        yA = self.resolutionBranch[0](xA)
        yD = []
        for i in range(len(xD)):
            yD_i = self.resolutionBranch[i + 1](xD[i])
            yD.append(yD_i)

        y = self.Decomposition_model.inv_transform(yA, yD)
        y = y.transpose(1, 2)
        xT = y[:, -self.pred_length:, :]  # decomposition output is always even, but pred length can be odd

        return xT


class WPMixer(nn.Module):
    """WPMixer模型：专门用于时间序列分类任务的模型

    Args:
        configs: 模型配置参数，包含seq_len, d_model, num_class等
            - seq_len: 输入序列长度
            - d_model: 模型的隐藏层维度
            - num_class: 分类任务的类别数
            - batch_size: 批处理大小
            - enc_in: 输入通道数（特征维度）
            - dropout: Dropout比率
            - device: 计算设备（如 'cpu' 或 'cuda'）
            - patch_len: 补丁长度，用于将时间序列分割成小块
            - use_amp: 是否使用自动混合精度训练
        tfactor: Token混合器的扩展因子，默认值为5
        dfactor: 嵌入混合器的扩展因子，默认值为5
        wavelet: 使用的小波基函数名称，默认值为'db2'
                - db2 , db3 , db5 等Daubechies小波 默认：db2
                - bior3.1 等Biorthogonal小波
                - sym3 , sym4 等Symlets小波
                - coif5 等Coiflets小波
        level: 小波分解的层级数，默认值为1
        stride: 补丁步长，控制补丁之间的重叠程度，默认值为8
        no_decomposition: 是否跳过小波分解步骤，默认值为False
    """

    def __init__(self, configs, tfactor=10, dfactor=5, wavelet='coif5', level=1, stride=8, no_decomposition=False):
        super(WPMixer, self).__init__()
        self.args = configs
        self.seq_len = configs.seq_len
        self.num_class = configs.num_class
        self.d_model = configs.d_model

        self.wpmixerCore = WPMixerCore(input_length=self.args.seq_len,
                                       pred_length=self.args.seq_len,  # 对于分类任务，预测长度等于序列长度
                                       wavelet_name=wavelet,
                                       level=level,
                                       batch_size=self.args.batch_size,
                                       channel=self.args.enc_in,  # 使用enc_in而不是c_out
                                       d_model=self.args.d_model,
                                       dropout=self.args.dropout,
                                       embedding_dropout=self.args.dropout,
                                       tfactor=tfactor,
                                       dfactor=dfactor,
                                       device=self.args.device,
                                       patch_len=self.args.patch_len,
                                       patch_stride=stride,
                                       no_decomposition=no_decomposition,
                                       use_amp=self.args.use_amp)

        # 分类任务专用投影层
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(
            configs.enc_in * configs.seq_len, configs.num_class)

    def forward(self, x_enc, x_mark_enc=None):
        """前向传播函数：处理输入序列并输出分类结果

        Args:
            x_enc: 输入序列数据，形状为 [B, T, C]
            x_mark_enc: 时间标记数据（可选），用于掩码处理，形状为 [B, T]

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        # WPMixer核心处理
        enc_out = self.wpmixerCore(x_enc)

        # 分类头处理
        output = self.act(enc_out)
        output = self.dropout(output)

        # 应用掩码（如果提供）
        if x_mark_enc is not None:
            output = output * x_mark_enc.unsqueeze(-1)

        # 展平特征并投影到类别空间
        output = output.reshape(output.shape[0], -1)  # [B, seq_len * enc_in]
        output = self.projection(output)  # [B, num_class]
        return output