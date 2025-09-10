# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class DLinear(nn.Module):
    """DLinear模型：专门用于时间序列分类任务的模型

    Args:
        configs: 模型配置参数，包含seq_len, enc_in, num_class等
        individual: Bool, 是否为不同变量使用独立的模型，默认为False
        表示不同变量间是否共享模型。
    """

    def __init__(self, configs, individual=True):
        """初始化DLinear分类模型

        Args:
            configs: 配置对象，包含模型参数
                - seq_len: 输入序列长度
                - enc_in: 输入特征维度
                - num_class: 分类类别数
                - moving_avg: 移动平均参数
            individual: 是否为每个变量使用独立的线性层
        """
        super(DLinear, self).__init__()
        self.seq_len: int = configs.seq_len
        self.pred_len: int= configs.seq_len  # 对于分类任务，预测长度等于序列长度
        # 来自Autoformer的序列分解模块
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in
        self.num_class = configs.num_class

        # 为季节性和趋势组件分别创建线性层
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                # 初始化权重
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # 初始化权重
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        # 分类任务专用投影层
        self.projection = nn.Linear(
            configs.enc_in * configs.seq_len, configs.num_class)
        # 激活函数和dropout层
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)

    def encoder(self, x):
        """编码器：对输入序列进行季节性和趋势分解，并通过线性层处理

        Args:
            x: 输入序列，形状为 [B, T, C]

        Returns:
            处理后的序列，形状为 [B, T, C]
        """
        # 序列分解
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)

        if self.individual:
            # 为每个通道使用独立的线性层
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            # 使用共享的线性层
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # 合并季节性和趋势组件
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forward(self, x_enc, x_mark_enc=None):
        """前向传播函数：处理输入序列并输出分类结果

        Args:
            x_enc: 输入序列数据，形状为 [B, T, C]
            x_mark_enc: 时间标记数据（可选），用于掩码处理

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        # 编码器处理
        enc_out = self.encoder(x_enc)

        # # 分类头处理
        # output = self.act(enc_out)
        # output = self.dropout(output)

        # # 应用掩码（如果提供）
        # if x_mark_enc is not None:
        #     output = output * x_mark_enc.unsqueeze(-1)

        # # 展平特征并投影到类别空间
        # output = output.reshape(output.shape[0], -1)  # [B, seq_len * enc_in]
        # output = self.projection(output)  # [B, num_class]


        # 展平特征并投影到类别空间
        output = enc_out.reshape(enc_out.shape[0], -1)  # [B, seq_len * enc_in]
        output = self.projection(output)  # [B, num_class]
        return output