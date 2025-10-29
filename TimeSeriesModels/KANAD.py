# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class KANANBlock(nn.Module):
    """KANANBlock模块：基于Kolmogorov-Arnold网络的分类特征提取模块

    使用周期性余弦函数和卷积操作提取时间序列的特征表示，
    专门优化用于分类任务的特征学习。

    Args:
        window: 输入序列长度
        order: 周期性函数的阶数
        d_model: 特征维度
    """

    def __init__(self, window: int, order: int, d_model: int) -> None:
        super().__init__()
        self.order = order
        self.window = window
        self.d_model = d_model
        self.channels = 2 * self.order + 1

        # 注册周期性余弦函数作为缓冲区
        self.register_buffer(
            "orders",
            self._create_custom_periodic_cosine(self.window, self.order).unsqueeze(0),
        )

        # 分类任务优化的卷积层
        self.out_conv = nn.Conv1d(self.channels, d_model, 1, bias=False)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.channels)
        self.bn3 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(self.channels)
        self.init_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.final_conv = nn.Linear(window, window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：提取时间序列的分类特征

        Args:
            x: 输入序列，形状为 [B, L]

        Returns:
            提取的特征，形状为 [B, d_model, L]
        """
        res = []
        res.append(x.unsqueeze(1))

        # 构建周期性特征
        ff = torch.concat(
            [self.orders.repeat(x.size(0), 1, 1)]
            + [torch.cos(order * x.unsqueeze(1)) for order in range(1, self.order + 1)]
            + [x.unsqueeze(1)],
            dim=1,
        )  # [B, channels, window]

        res.append(ff)

        # 特征提取流程
        ff = self.init_conv(ff)
        ff = self.bn1(ff)
        ff = self.act(ff)

        ff = self.inner_conv(ff) + res.pop()
        ff = self.bn2(ff)
        ff = self.act(ff)

        ff = self.out_conv(ff) + res.pop()
        ff = self.bn3(ff)
        ff = self.act(ff)

        ff = self.final_conv(ff)

        return ff

    def _create_custom_periodic_cosine(self, window: int, period) -> torch.Tensor:
        """创建自定义周期性余弦函数

        Args:
            window: 序列长度
            period: 周期参数

        Returns:
            周期性余弦函数值
        """
        d = len(period) if isinstance(period, list) else period
        pl = period if isinstance(period, list) else [i for i in range(1, period + 1)]
        result = torch.empty(d, window, dtype=torch.float32)
        for i, p in enumerate(pl):
            t = torch.arange(0, 1, 1 / window, dtype=torch.float32) / p * 2 * np.pi
            result[i, :] = torch.cos(t)
        return result


class KANAD(nn.Module):
    """KANAN模型：专门用于时间序列分类任务的Kolmogorov-Arnold网络

    基于KANAD模型重构，移除了异常检测等功能，专注于分类任务。
    使用周期性函数和卷积操作提取时间序列特征进行分类。

    Args:
        configs: 模型配置参数，包含seq_len, d_model, num_class等
    """

    def __init__(self, configs):
        super(KANAD, self).__init__()
        self.configs = configs
        self.enc_in = configs.enc_in
        self.seq_len:int = configs.seq_len
        self.num_class: int = configs.num_class
        self.d_model: int = configs.d_model

        # 特征提取器
        self.feature_extractor = KANANBlock(
            window=self.seq_len,
            order=configs.d_model,
            d_model=configs.d_model
        )

        # 分类头
        self.act = nn.GELU()
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len*self.enc_in, configs.num_class)
        self.layer_norm = nn.LayerNorm(configs.d_model)

    def forward(self, x_enc, x_mark_enc=None):
        """前向传播：处理输入序列并输出分类结果

        Args:
            x_enc: 输入序列数据，形状为 [B, T, C]
            x_mark_enc: 时间标记数据，用于掩码填充部分，形状为 [B, T]

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        # 处理多变量时间序列 [B, T, C] -> [B * C, T]
        B, T, C = x_enc.size()
        x_input = rearrange(x_enc, "B T C -> (B C) T")

        # 特征提取 [B * C, T] -> [B * C, d_model, T]
        features = self.feature_extractor(x_input)

        # 特征聚合和规范化
        features = self.layer_norm(features.permute(0, 2, 1)).permute(0, 2, 1)

        # 重塑回原始批次维度 [B * C, d_model, T] -> [B, C * d_model * T]
        features = rearrange(features, "(B C) D T -> B (C D T)", B=B, C=C)

        # 分类头处理
        output = self.act(features)
        output = self.dropout(output)

        # 应用掩码（如果提供）
        if x_mark_enc is not None:
            # 扩展掩码以匹配特征维度
            mask = x_mark_enc.unsqueeze(-1).repeat(1, 1, self.d_model)
            mask = rearrange(mask, "B T D -> B (T D)")
            output = output * mask

        # 投影到类别空间
        output = self.projection(output)  # [B, num_class]
        return output