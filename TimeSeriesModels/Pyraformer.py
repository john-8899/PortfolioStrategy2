# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from layers.Pyraformer_EncDec import Encoder


class Pyraformer(nn.Module):
    """
    Pyraformer: 专门用于时间序列分类任务的Pyraformer模型

    基于金字塔注意力机制，通过多尺度特征提取和注意力机制来捕获时间序列的
    长期依赖关系和局部特征，专为分类任务优化设计。

    Paper link: https://openreview.net/pdf?id=0EXmFzUn5I

    Args:
        configs: 模型配置参数，包含seq_len, d_model, num_class等
        window_size: list, 金字塔注意力中的下采样窗口大小
        作用: 控制在金字塔结构中每层的下采样比例，用于捕获多尺度的时间序列特征
            参考范围: [2, 8] 的整数组合
            常见设置:
            [4, 4] (默认值) - 两层金字塔结构，每层都以4倍比例下采样
            [2, 4] - 较小的第一层下采样比例，保留更多细节
            [4, 8] - 更大幅度的下采样，适合长序列
            选择建议:
            序列较短时使用较小值(如 [2, 2])
            序列较长时可使用较大值(如 [4, 8])
            需要平衡计算效率和特征表达能力
        inner_size: int, 邻域注意力的大小
        作用: 定义在局部邻域内进行注意力计算的范围，用于捕获局部特征和依赖关系
            参考范围: 3-10
            常见设置:
            5 (默认值) - 中等邻域大小，平衡局部和全局信息
            3 - 更关注局部细节特征
            7 或 9 - 考虑更大范围的邻域依赖
            选择建议:
            数据变化剧烈时使用较小值
            需要捕获长距离局部依赖时使用较大值
    """

    def __init__(self, configs, window_size=[2, 4], inner_size=5):
        """
        初始化Pyraformer2模型

        Args:
            configs: 模型配置参数
            window_size: 下采样窗口大小列表，默认为[4, 4]
            inner_size: 邻域注意力大小，默认为5
        """
        super().__init__()
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.num_class = configs.num_class

        # 初始化金字塔编码器
        self.encoder = Encoder(configs, window_size, inner_size)

        # 分类任务专用层
        self.act = torch.nn.functional.gelu
        self.dropout = nn.Dropout(configs.dropout)

        # 分类投影层：将序列特征映射到类别空间
        self.projection = nn.Linear(
            (len(window_size)+1)*configs.d_model * configs.seq_len, configs.num_class
        )

    def forward(self, x_enc, x_mark_enc=None):
        """
        前向传播函数：处理输入序列并输出分类结果

        Args:
            x_enc: 输入序列数据，形状为 [B, T, C]
            x_mark_enc: 时间标记数据，用于掩码填充部分，形状为 [B, T]

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        # 通过金字塔编码器提取多尺度特征
        enc_out = self.encoder(x_enc, x_mark_enc)

        # 分类头处理
        output = self.act(enc_out)
        output = self.dropout(output)

        # 应用掩码（如果提供）- 零填充嵌入
        if x_mark_enc is not None:
            output = output * x_mark_enc.unsqueeze(-1)

        # 展平特征：将序列维度展平
        output = output.reshape(output.shape[0], -1)  # [B, seq_len * d_model]

        # 投影到类别空间
        output = self.projection(output)  # [B, num_class]

        return output