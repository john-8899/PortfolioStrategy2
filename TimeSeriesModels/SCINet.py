# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Splitting(nn.Module):
    """数据分割模块：将输入序列分割为偶数和奇数部分

    Args:
        无参数，仅实现基本的分割功能
    """

    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        """提取偶数索引位置的数据

        Args:
            x: 输入张量，形状为 [B, T, C]

        Returns:
            偶数位置的数据，形状为 [B, T//2, C]
        """
        return x[:, ::2, :]

    def odd(self, x):
        """提取奇数索引位置的数据

        Args:
            x: 输入张量，形状为 [B, T, C]

        Returns:
            奇数位置的数据，形状为 [B, T//2, C]
        """
        return x[:, 1::2, :]

    def forward(self, x):
        """前向传播：返回分割后的偶数和奇数部分

        Args:
            x: 输入张量，形状为 [B, T, C]

        Returns:
            tuple: (偶数部分, 奇数部分)，形状均为 [B, T//2, C]
        """
        return self.even(x), self.odd(x)


class CausalConvBlock(nn.Module):
    """因果卷积块：使用因果卷积处理时间序列数据

    Args:
        d_model: 输入和输出的特征维度
        kernel_size: 卷积核大小，默认为5
        dropout: Dropout概率，默认为0.0
    """

    def __init__(self, d_model, kernel_size=5, dropout=0.0):
        super(CausalConvBlock, self).__init__()
        module_list = [
            nn.ReplicationPad1d((kernel_size - 1, kernel_size - 1)),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size),
            nn.Tanh()
        ]
        self.causal_conv = nn.Sequential(*module_list)

    def forward(self, x):
        """前向传播：应用因果卷积

        Args:
            x: 输入张量，形状为 [B, C, T]

        Returns:
            卷积后的输出，形状为 [B, C, T]
        """
        return self.causal_conv(x)


class SCIBlock(nn.Module):
    """SCI（Sample-Causal-Interaction）块：核心交互模块

    Args:
        d_model: 特征维度
        kernel_size: 卷积核大小，默认为5
        dropout: Dropout概率，默认为0.0
    """

    def __init__(self, d_model, kernel_size=5, dropout=0.0):
        super(SCIBlock, self).__init__()
        self.splitting = Splitting()
        # 创建四个因果卷积块用于特征交互
        self.modules_even, self.modules_odd, self.interactor_even, self.interactor_odd = [
            CausalConvBlock(d_model, kernel_size, dropout) for _ in range(4)]

    def forward(self, x):
        """前向传播：执行样本-因果交互

        Args:
            x: 输入张量，形状为 [B, T, C]

        Returns:
            tuple: (更新后的偶数部分, 更新后的奇数部分)
        """
        # 分割输入为偶数和奇数部分
        x_even, x_odd = self.splitting(x)

        # 调整维度以适配卷积操作 [B, T, C] -> [B, C, T]
        x_even = x_even.permute(0, 2, 1)
        x_odd = x_odd.permute(0, 2, 1)

        # 交互计算：使用指数函数增强特征交互
        x_even_temp = x_even.mul(torch.exp(self.modules_even(x_odd)))
        x_odd_temp = x_odd.mul(torch.exp(self.modules_odd(x_even)))

        # 更新特征
        x_even_update = x_even_temp + self.interactor_even(x_odd_temp)
        x_odd_update = x_odd_temp - self.interactor_odd(x_even_temp)

        # 调整维度回原始格式 [B, C, T] -> [B, T, C]
        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1)


class SCINet(nn.Module):
    """SCINet主干网络：递归构建的树状结构

    Args:
        d_model: 特征维度
        current_level: 当前递归层级，默认为3
        kernel_size: 卷积核大小，默认为5
        dropout: Dropout概率，默认为0.0
    """

    def __init__(self, d_model, current_level=3, kernel_size=5, dropout=0.0):
        super(SCINet, self).__init__()
        self.current_level = current_level
        self.working_block = SCIBlock(d_model, kernel_size, dropout)

        # 递归构建树状结构
        if current_level != 0:
            self.SCINet_Tree_odd = SCINet(d_model, current_level - 1, kernel_size, dropout)
            self.SCINet_Tree_even = SCINet(d_model, current_level - 1, kernel_size, dropout)

    def forward(self, x):
        """前向传播：处理输入序列

        Args:
            x: 输入张量，形状为 [B, T, C]

        Returns:
            处理后的输出，形状为 [B, T, C]
        """
        # 处理奇数长度序列
        odd_flag = False
        if x.shape[1] % 2 == 1:
            odd_flag = True
            x = torch.cat((x, x[:, -1:, :]), dim=1)

        # 通过SCI块处理
        x_even_update, x_odd_update = self.working_block(x)
        if odd_flag:
            x_odd_update = x_odd_update[:, :-1]

        # 递归处理或合并结果
        if self.current_level == 0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(
                self.SCINet_Tree_even(x_even_update),
                self.SCINet_Tree_odd(x_odd_update)
            )

    def zip_up_the_pants(self, even, odd):
        """合并偶数和奇数部分

        Args:
            even: 偶数部分特征，形状为 [B, T_even, C]
            odd: 奇数部分特征，形状为 [B, T_odd, C]

        Returns:
            合并后的完整序列，形状为 [B, T, C]
        """
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2)
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        min_len = min(even_len, odd_len)

        # 交替合并偶数和奇数部分
        zipped_data = []
        for i in range(min_len):
            zipped_data.append(even[i].unsqueeze(0))
            zipped_data.append(odd[i].unsqueeze(0))

        # 处理长度不匹配的情况
        if even_len > odd_len:
            zipped_data.append(even[-1].unsqueeze(0))

        return torch.cat(zipped_data, 0).permute(1, 0, 2)


class SCINetModel(nn.Module):
    """SCINet模型：专门用于时间序列分类任务的优化版本

    Args:
        configs: 模型配置参数，包含enc_in, seq_len, num_class等
    """

    def __init__(self, configs):
        super(SCINetModel, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.num_class = configs.num_class

        # SCINet主干网络
        self.sci_net = SCINet(d_model=configs.enc_in, current_level=configs.current_level, kernel_size=configs.kernel_size,
                              dropout=configs.dropout)

        # 分类专用投影层
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)

    def forward(self, x_enc, x_mark_enc=None):
        """前向传播：处理输入序列并输出分类结果

        Args:
            x_enc: 输入序列数据，形状为 [B, T, C]
            x_mark_enc: 时间标记数据（可选），用于掩码填充部分

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        # 通过SCINet主干网络提取特征
        enc_out = self.sci_net(x_enc)  # [B, T, C]

        # 残差连接增强特征表达
        enc_out = enc_out + x_enc

        # 分类头处理
        output = self.act(enc_out)
        output = self.dropout(output)

        # 应用掩码（如果提供时间标记）
        if x_mark_enc is not None:
            output = output * x_mark_enc.unsqueeze(-1)

        # 展平特征并投影到类别空间
        output = output.reshape(output.shape[0], -1)  # [B, T * C]
        output = self.projection(output)  # [B, num_class]

        return output