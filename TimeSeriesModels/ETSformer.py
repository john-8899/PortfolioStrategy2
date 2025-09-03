# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
from layers.ETSformer_EncDec import EncoderLayer, Encoder


class ETSformer(nn.Module):
    """ETSformer模型：专门用于时间序列分类任务的模型

    Paper link: https://arxiv.org/abs/2202.01381

    Attributes:
        seq_len (int): 输入序列长度
        enc_embedding (DataEmbedding): 数据嵌入层
        encoder (Encoder): ETSformer编码器
        act (function): 激活函数(gelu)
        dropout (nn.Dropout): Dropout层
        projection (nn.Linear): 分类投影层
    """

    def __init__(self, configs):
        """初始化ETSformer2模型

        Args:
            configs: 模型配置参数对象，包含以下属性：
                - seq_len (int): 输入序列长度，表示时间序列的长度
                - d_model (int): 模型维度，表示嵌入向量的维度
                - num_class (int): 分类数量，表示最终分类任务的类别数
                - enc_in (int): 编码器输入特征数，表示输入序列的特征维度
                - embed (str): 嵌入类型，指定使用的嵌入方式（如'timeF'）
                - freq (str): 时间频率，表示时间序列数据的时间粒度（如'h'表示小时）
                - dropout (float): Dropout比率，用于防止过拟合
                - n_heads (int): 注意力头数，表示多头注意力机制中的头数
                - d_ff (int): 前馈网络维度，表示前馈网络中的隐藏层维度
                - activation (str): 激活函数类型，指定使用的激活函数（如'sigmoid'）
                - e_layers (int): 编码器层数，表示编码器中堆叠的层数
                - top_k (int): Top K个显著周期，用于傅里叶变换中选择主要频率成分
        """
        super(ETSformer, self).__init__()
        self.seq_len = configs.seq_len

        # 数据嵌入层，将输入序列转换为高维表示
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # ETSformer编码器，用于提取时间序列特征
        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.d_model, configs.n_heads, configs.enc_in, configs.seq_len, configs.seq_len, configs.top_k,
                    dim_feedforward=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ]
        )

        # 分类任务专用投影层
        self.act = torch.nn.functional.gelu  # GELU激活函数
        self.dropout = nn.Dropout(configs.dropout)  # Dropout层，防止过拟合
        # 将序列表示投影到类别空间
        self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forward(self, x_enc, x_mark_enc=None):
        """前向传播函数：处理输入序列并输出分类结果

        Args:
            x_enc (Tensor): 输入序列数据，形状为 [B, T, C]
                - B: batch size，表示批次大小
                - T: seq_len，表示序列长度
                - C: enc_in，表示输入特征数
            x_mark_enc (Tensor, optional): 时间标记数据，用于掩码填充部分，形状为 [B, T]

        Returns:
            output (Tensor): 分类结果，形状为 [B, num_class]
                - B: batch size，表示批次大小
                - num_class: 分类数量，表示最终分类任务的类别数
        """
        # 数据嵌入，将输入序列转换为高维表示
        enc_out = self.enc_embedding(x_enc, None)

        # 通过编码器提取特征
        _, growths, seasons = self.encoder(enc_out, x_enc, attn_mask=None)

        # 聚合growth和season成分
        growths = torch.sum(torch.stack(growths, 0), 0)[:, :self.seq_len, :]
        seasons = torch.sum(torch.stack(seasons, 0), 0)[:, :self.seq_len, :]
        enc_out = growths + seasons

        # 应用激活函数和Dropout
        output = self.act(enc_out)  # GELU激活
        output = self.dropout(output)  # Dropout正则化

        # 应用掩码（如果提供）
        if x_mark_enc is not None:
            output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings

        # 展平特征并投影到类别空间
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output