# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """残差块模块：包含时间维度和通道维度的混合操作

    该模块通过两个独立的序列处理时间维度和通道维度，
    并使用残差连接保留原始信息，增强特征提取能力。

    Args:
        configs: 模型配置参数，包含seq_len, d_model, enc_in, dropout等
    """

    def __init__(self, configs):
        super(ResBlock, self).__init__()

        # 时间维度处理序列
        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len),
            nn.Dropout(configs.dropout)
        )

        # 通道维度处理序列
        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        """前向传播函数

        Args:
            x: 输入张量，形状为 [B, L, D]，其中B是批次大小，L是序列长度，D是特征维度

        Returns:
            output: 处理后的张量，形状与输入相同 [B, L, D]
        """
        # 时间维度残差连接：先转置使时间维度位于特征维度，处理后转置回来
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        # 通道维度残差连接
        x = x + self.channel(x)
        return x


class TSMixer(nn.Module):
    """TSMixer模型：专门用于时间序列分类任务的模型

    该模型基于TSMixer架构，但专门针对分类任务进行了优化。
    通过堆叠的残差块进行特征提取，并使用专门的分类头进行分类。

    Args:
        configs: 模型配置参数，包含seq_len, enc_in, d_model, e_layers, dropout, num_class等
    """

    def __init__(self, configs):
        super(TSMixer, self).__init__()

        self.seq_len = configs.seq_len
        self.num_class = configs.num_class
        self.d_model = configs.d_model
        self.layer = configs.e_layers

        # 创建堆叠的残差块
        self.res_blocks = nn.ModuleList([ResBlock(configs) for _ in range(configs.e_layers)])

        # 分类任务专用组件
        self.act = F.gelu  # 激活函数
        self.dropout = nn.Dropout(configs.dropout)  # Dropout层

        # 分类投影层：将提取的特征映射到类别空间
        self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)


    def classification(self, x_enc, x_mark_enc=None):
        """分类任务的前向传播函数

        Args:
            x_enc: 输入序列数据，形状为 [B, L, D]
            x_mark_enc: 可选的时间标记，用于掩码处理，形状为 [B, L]

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        # 通过残差块堆叠进行特征提取
        for i in range(self.layer):
            x_enc = self.res_blocks[i](x_enc)

        # 应用激活函数和dropout
        output = self.act(x_enc)
        output = self.dropout(output)

        # 如果提供了时间标记，应用掩码（处理变长序列）
        if x_mark_enc is not None:
            output = output * x_mark_enc.unsqueeze(-1)

        # 展平特征并投影到类别空间
        output = output.reshape(output.shape[0], -1)  # [B, seq_len * d_model]
        output = self.projection(output)  # [B, num_class]

        return output

    def forward(self, x_enc, x_mark_enc=None):
        """模型的前向传播函数

        Args:
            x_enc: 输入序列数据，形状为 [B, L, D]
            x_mark_enc: 可选的时间标记，用于掩码处理，形状为 [B, L]

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        # 直接调用分类函数
        return self.classification(x_enc, x_mark_enc)