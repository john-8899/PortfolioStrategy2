# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class IEBlock(nn.Module):
    """IEBlock模块：用于特征提取和变换

    Args:
        input_dim: 输入维度
        hid_dim: 隐藏层维度
        output_dim: 输出维度
        num_node: 节点数量
    """

    def __init__(self, input_dim, hid_dim, output_dim, num_node):
        super(IEBlock, self).__init__()

        self.input_dim:int = input_dim
        self.hid_dim: int = hid_dim
        self.output_dim:int = output_dim
        self.num_node: int = num_node

        self._build()

    def _build(self):
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim // 4)
        )

        self.channel_proj = nn.Linear(self.num_node, self.num_node)
        torch.nn.init.eye_(self.channel_proj.weight)

        self.output_proj = nn.Linear(self.hid_dim // 4, self.output_dim)

    def forward(self, x):
        x = self.spatial_proj(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1) + self.channel_proj(x.permute(0, 2, 1))
        x = self.output_proj(x.permute(0, 2, 1))

        x = x.permute(0, 2, 1)

        return x


class LightTS(nn.Module):
    """LightTS模型：专门用于时间序列分类任务的模型
     Paper link: https://arxiv.org/abs/2207.01186
    Args:
        configs: 模型配置参数，包含seq_len, d_model, num_class等
    """

    def __init__(self, configs):
        super(LightTS, self).__init__()
        self.seq_len: int = configs.seq_len
        #chunk_size: int, reshape T into [num_chunks, chunk_size]
        self.chunk_size = min(configs.seq_len, configs.chunk_size)

        # padding in order to ensure complete division
        # 检查seq_len是否能被chunk_size整除
        # 如果不能整除，则增加seq_len使其能被整除
        # 计算并存储块的数量
        if self.seq_len % self.chunk_size != 0:
            self.seq_len += (self.chunk_size - self.seq_len % self.chunk_size)
        self.num_chunks = self.seq_len // self.chunk_size

        self.d_model: int = configs.d_model
        self.enc_in: int = configs.enc_in
        self.num_class: int = configs.num_class

        # 分类任务专用投影层
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(self.enc_in * self.seq_len, configs.num_class)

        self._build()

    def _build(self):
        self.layer_1 = IEBlock(
            input_dim=self.chunk_size,
            hid_dim=self.d_model // 4,
            output_dim=self.d_model // 4,
            num_node=self.num_chunks
        )

        self.chunk_proj_1 = nn.Linear(self.num_chunks, 1)

        self.layer_2 = IEBlock(
            input_dim=self.chunk_size,
            hid_dim=self.d_model // 4,
            output_dim=self.d_model // 4,
            num_node=self.num_chunks
        )

        self.chunk_proj_2 = nn.Linear(self.num_chunks, 1)

        self.layer_3 = IEBlock(
            input_dim=self.d_model // 2,
            hid_dim=self.d_model // 2,
            output_dim=self.seq_len,  # 修改为seq_len以适应分类任务
            num_node=self.enc_in
        )

        self.ar = nn.Linear(self.seq_len, self.seq_len)  # 修改为seq_len以适应分类任务

    def encoder(self, x):
        """编码器：处理输入序列并提取特征

        Args:
            x: 输入序列数据，形状为 [B, T, N]

        Returns:
            out: 编码后的特征，形状为 [B, seq_len, N]
        """
        B, T, N = x.size()

        # padding
        x = torch.cat([x, torch.zeros((B, self.seq_len - T, N)).to(x.device)], dim=1)

        highway = self.ar(x.permute(0, 2, 1))
        highway = highway.permute(0, 2, 1)

        # continuous sampling
        x1 = x.reshape(B, self.num_chunks, self.chunk_size, N)
        x1 = x1.permute(0, 3, 2, 1)
        x1 = x1.reshape(-1, self.chunk_size, self.num_chunks)
        x1 = self.layer_1(x1)
        x1 = self.chunk_proj_1(x1).squeeze(dim=-1)

        # interval sampling
        x2 = x.reshape(B, self.chunk_size, self.num_chunks, N)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = x2.reshape(-1, self.chunk_size, self.num_chunks)
        x2 = self.layer_2(x2)
        x2 = self.chunk_proj_2(x2).squeeze(dim=-1)

        x3 = torch.cat([x1, x2], dim=-1)

        x3 = x3.reshape(B, N, -1)
        x3 = x3.permute(0, 2, 1)

        out = self.layer_3(x3)

        out = out + highway
        return out

    def forward(self, x_enc, x_mark_enc=None):
        """前向传播函数：处理输入序列并输出分类结果

        Args:
            x_enc: 输入序列数据，形状为 [B, T, N]
            x_mark_enc: 时间标记数据，用于掩码填充部分，形状为 [B, T]

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        enc_out = self.encoder(x_enc)

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