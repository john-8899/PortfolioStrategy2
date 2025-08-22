# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class ResBlock(nn.Module):
    """残差块：用于构建TiDE模型的基础组件

    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        dropout: Dropout比率
        bias: 是否使用偏置
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = LayerNorm(output_dim, bias=bias)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)
        out = self.ln(out)
        return out


class TiDE(nn.Module):
    """TiDE模型：专门用于时间序列分类任务的模型
    paper: https://arxiv.org/pdf/2304.08424.pdf
    Args:
        configs: 模型配置参数，包含seq_len, d_model, num_class等
    """

    def __init__(self, configs ):
        """初始化TiDE分类模型

        Args:
            configs: 模型配置参数
        """
        super(TiDE, self).__init__()
        self.configs = configs
        self.enc_in: int = configs.enc_in  # 输入特征维度(特征数)
        self.seq_len: int = configs.seq_len  # 序列长度
        self.hidden_dim: int = configs.d_model  # 隐藏层维度
        self.res_hidden: int = configs.d_model  # 残差块隐藏层维度
        self.encoder_num: int = configs.e_layers  # 编码器层数
        self.bias: bool = configs.bias # 是否在Linear层中使用偏置，默认为True
        self.feature_encode_dim: int = configs.feature_encode_dim  # 特征编码维度
        self.num_class: int = configs.num_class  # 分类任务的类别数
        dropout = configs.dropout  # Dropout比率

        # 特征编码器
        self.feature_encoder = ResBlock(self.seq_len, self.res_hidden, self.feature_encode_dim, dropout, self.bias)

        # 编码器堆叠
        flatten_dim = self.enc_in*(1 + self.feature_encode_dim) # 特征编码器的输出维度
        self.encoders = nn.Sequential(
            ResBlock(flatten_dim, self.res_hidden, self.hidden_dim, dropout, self.bias),
            *([ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, self.bias)] * (self.encoder_num - 1))
        )

        # 分类任务专用投影层
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(self.hidden_dim, self.num_class)

    def forward(self, x_enc, x_mark_enc=None):
        """前向传播函数：处理输入序列并输出分类结果

        Args:
            x_enc: 输入序列数据，形状为 [B, T, C]
            x_mark_enc: 时间标记数据，用于掩码填充部分，形状为 [B, T]

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        B, T, C = x_enc.shape  # [32, 60, 87]
        # 特征编码
        feature = self.feature_encoder(x_enc.permute(0, 2, 1))
        # print(f"feature size {feature.shape}")
        # print(f"x_enc size {x_enc.shape}")

        # 对时间维度进行全局平均池化，得到每个通道的表示
        # [B, T, C] -> [B, C] (通过时间维度的平均池化)
        x_pooled = torch.mean(x_enc, dim=1)  # [B, C]

        # 拼接原始通道特征和编码特征
        # [B, C] + [B, C, feature_encode_dim] -> [B, C*(1+feature_encode_dim)]
        feature_flat = feature.reshape(B, -1)  # [B, C*feature_encode_dim]
        # print(f"feature_flat size {feature_flat.shape} ,xpooled size {x_pooled.shape}")
        # # 编码器处理
        hidden = self.encoders(torch.cat([x_pooled, feature_flat], dim=-1))

        # 分类头处理
        output = self.act(hidden)
        output = self.dropout(output)

        # 投影到类别空间
        output = self.projection(output)  # [B, num_class]
        return output