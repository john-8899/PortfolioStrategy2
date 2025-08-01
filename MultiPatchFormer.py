# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
from einops import rearrange

from layers.SelfAttention_Family import AttentionLayer, FullAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 512):
        super(FeedForward, self).__init__()

        self.linear_1 = torch.nn.Linear(d_model, d_hidden)
        self.linear_2 = torch.nn.Linear(d_hidden, d_model)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)

        return x


class Encoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            mha: AttentionLayer,
            d_hidden: int,
            dropout: float = 0,
            channel_wise=False,
    ):
        super(Encoder, self).__init__()

        self.channel_wise = channel_wise
        if self.channel_wise:
            self.conv = torch.nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="reflect",
            )
        self.MHA = mha
        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        q = residual
        if self.channel_wise:
            x_r = self.conv(x.permute(0, 2, 1)).transpose(1, 2)
            k = x_r
            v = x_r
        else:
            k = residual
            v = residual
        x, score = self.MHA(q, k, v, attn_mask=None)
        x = self.dropout(x)
        x = self.layerNormal_1(x + residual)

        residual = x
        x = self.feedforward(residual)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)

        return x, score


class MultiPatchFormer(nn.Module):
    """
    MultiPatchFormer分类模型
    基于MultiPatchFormer架构，专为时间序列分类任务设计
    """

    def __init__(self, configs):
        super(MultiPatchFormer, self).__init__()
        self.seq_len = configs.seq_len
        self.num_class: int = configs.num_class
        self.d_channel: int = configs.enc_in
        self.N = configs.e_layers
        # Embedding
        self.d_model: int = configs.d_model
        self.d_hidden: int = configs.d_ff
        self.n_heads = configs.n_heads
        self.mask = True
        self.dropout: float = configs.dropout

        self.stride1 = 8
        self.patch_len1 = 8
        self.stride2 = 8
        self.patch_len2 = 16
        self.stride3 = 7
        self.patch_len3 = 24
        self.stride4 = 6
        self.patch_len4 = 32
        self.patch_num1 = int((self.seq_len - self.patch_len2) // self.stride2) + 2
        self.padding_patch_layer1 = nn.ReplicationPad1d((0, self.stride1))
        self.padding_patch_layer2 = nn.ReplicationPad1d((0, self.stride2))
        self.padding_patch_layer3 = nn.ReplicationPad1d((0, self.stride3))
        self.padding_patch_layer4 = nn.ReplicationPad1d((0, self.stride4))

        self.shared_MHA = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(mask_flag=self.mask),
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                )
                for _ in range(self.N)
            ]
        )

        self.shared_MHA_ch = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(mask_flag=self.mask),
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                )
                for _ in range(self.N)
            ]
        )

        self.encoder_list = nn.ModuleList(
            [
                Encoder(
                    d_model=self.d_model,
                    mha=self.shared_MHA[ll], #type:ignore
                    d_hidden=self.d_hidden,
                    dropout=self.dropout,
                    channel_wise=False,
                )
                for ll in range(self.N)
            ]
        )

        self.encoder_list_ch = nn.ModuleList(
            [
                Encoder(
                    d_model=self.d_model,
                    mha=self.shared_MHA_ch[0],#type:ignore
                    d_hidden=self.d_hidden,
                    dropout=self.dropout,
                    channel_wise=True,
                )
                for _ in range(self.N)
            ]
        )

        pe = torch.zeros(self.patch_num1, self.d_model)
        for pos in range(self.patch_num1):
            for i in range(0, self.d_model, 2):
                wavelength = 10000 ** ((2 * i) / self.d_model)
                pe[pos, i] = math.sin(pos / wavelength)
                pe[pos, i + 1] = math.cos(pos / wavelength)
        pe = pe.unsqueeze(0)  # add a batch dimention to your pe matrix
        self.register_buffer("pe", pe)

        self.embedding_channel = nn.Conv1d(
            in_channels=self.d_model * self.patch_num1,
            out_channels=self.d_model,
            kernel_size=1,
        )

        self.embedding_patch_1 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len1,
            stride=self.stride1,
        )
        self.embedding_patch_2 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len2,
            stride=self.stride2,
        )
        self.embedding_patch_3 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len3,
            stride=self.stride3,
        )
        self.embedding_patch_4 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len4,
            stride=self.stride4,
        )

        # 分类任务专用投影层
        self.act = torch.nn.GELU()
        self.dropout_layer = torch.nn.Dropout(p=self.dropout)
        self.projection = nn.Linear(self.d_model * self.d_channel, self.num_class)

    def forward(self, x_enc, x_mark_enc=None):
        """
        前向传播函数：处理输入序列并输出分类结果

        Args:
            x_enc: 输入序列数据，形状为 [B, T, C]
            x_mark_enc: 时间标记数据，用于掩码填充部分，形状为 [B, T]

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        # Multi-scale embedding
        x_i = x_enc.permute(0, 2, 1)

        x_i_p1 = x_i
        x_i_p2 = self.padding_patch_layer2(x_i)
        x_i_p3 = self.padding_patch_layer3(x_i)
        x_i_p4 = self.padding_patch_layer4(x_i)
        encoding_patch1 = self.embedding_patch_1(
            rearrange(x_i_p1, "b c l -> (b c) l").unsqueeze(-1).permute(0, 2, 1)
        ).permute(0, 2, 1)
        encoding_patch2 = self.embedding_patch_2(
            rearrange(x_i_p2, "b c l -> (b c) l").unsqueeze(-1).permute(0, 2, 1)
        ).permute(0, 2, 1)
        encoding_patch3 = self.embedding_patch_3(
            rearrange(x_i_p3, "b c l -> (b c) l").unsqueeze(-1).permute(0, 2, 1)
        ).permute(0, 2, 1)
        encoding_patch4 = self.embedding_patch_4(
            rearrange(x_i_p4, "b c l -> (b c) l").unsqueeze(-1).permute(0, 2, 1)
        ).permute(0, 2, 1)

        encoding_patch = (
                torch.cat(
                    (encoding_patch1, encoding_patch2, encoding_patch3, encoding_patch4),
                    dim=-1,
                )
                + self.pe
        )
        # Temporal encoding
        for i in range(self.N):
            encoding_patch = self.encoder_list[i](encoding_patch)[0]

        # Channel-wise encoding
        x_patch_c = rearrange(
            encoding_patch, "(b c) p d -> b c (p d)", b=x_enc.shape[0], c=self.d_channel
        )
        x_ch = self.embedding_channel(x_patch_c.permute(0, 2, 1)).transpose(
            1, 2
        )  # [b c d]

        encoding_1_ch = self.encoder_list_ch[0](x_ch)[0]

        # 分类头处理
        output = self.act(encoding_1_ch)
        output = self.dropout_layer(output)

        # 展平特征并投影到类别空间
        output = output.reshape(output.shape[0], -1)  # [B, d_model * d_channel]
        output = self.projection(output)  # [B, num_class]

        return output