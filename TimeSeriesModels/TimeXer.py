# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding


class FlattenHead(nn.Module):
    """扁平化头部，用于将多维特征展平并映射到目标维度"""

    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        """
        初始化扁平化头部

        Args:
            n_vars (int): 变量数量
            nf (int): 输入特征维度
            target_window (int): 目标窗口大小
            head_dropout (float): Dropout概率，默认为0
        """
        super().__init__()
        self.n_vars = n_vars  # 变量数量
        self.flatten = nn.Flatten(start_dim=-2)  # 扁平化层
        self.linear = nn.Linear(nf, target_window)  # 线性映射层
        self.dropout = nn.Dropout(head_dropout)  # Dropout层

    def forward(self, x):
        """
        前向传播函数

        Args:
            x (Tensor): 输入张量，形状为 [bs x nvars x d_model x patch_num]

        Returns:
            Tensor: 输出张量
        """
        # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    """编码器嵌入层，用于处理输入序列的嵌入表示"""

    def __init__(self, n_vars, d_model, patch_len, dropout):
        """
        初始化编码器嵌入层

        Args:
            n_vars (int): 变量数量
            d_model (int): 模型维度
            patch_len (int): patch长度
            dropout (float): Dropout概率
        """
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len  # patch长度

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)  # 值嵌入层
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))  # 全局token参数
        self.position_embedding = PositionalEmbedding(d_model)  # 位置嵌入层

        self.dropout = nn.Dropout(dropout)  # Dropout层

    def forward(self, x):
        """
        前向传播函数

        Args:
            x (Tensor): 输入张量

        Returns:
            Tuple[Tensor, int]: 嵌入后的张量和变量数量
        """
        # do patching
        n_vars = x.shape[1]  # 变量数量
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))  # 重复全局token以匹配批次大小

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)  # 对输入进行patch操作
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # 重塑张量形状
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)  # 值嵌入和位置嵌入相加
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))  # 重塑张量形状
        x = torch.cat([x, glb], dim=2)  # 将全局token连接到嵌入序列
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # 重塑张量形状
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    """编码器，包含多个编码器层"""

    def __init__(self, layers, norm_layer=None, projection=None):
        """
        初始化编码器

        Args:
            layers (list): 编码器层列表
            norm_layer (nn.Module, optional): 归一化层，默认为None
            projection (nn.Module, optional): 投影层，默认为None
        """
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)  # 编码器层列表
        self.norm = norm_layer  # 归一化层
        self.projection = projection  # 投影层

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        前向传播函数

        Args:
            x (Tensor): 输入张量
            cross (Tensor): 交叉注意力输入张量
            x_mask (Tensor, optional): 输入掩码，默认为None
            cross_mask (Tensor, optional): 交叉注意力掩码，默认为None
            tau (Tensor, optional): 时间衰减因子，默认为None
            delta (Tensor, optional): 时间间隔，默认为None

        Returns:
            Tensor: 编码后的张量
        """
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    """编码器层，包含自注意力和交叉注意力机制"""

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        """
        初始化编码器层

        Args:
            self_attention (nn.Module): 自注意力层
            cross_attention (nn.Module): 交叉注意力层
            d_model (int): 模型维度
            d_ff (int, optional): 前馈网络维度，默认为4*d_model
            dropout (float): Dropout概率，默认为0.1
            activation (str): 激活函数类型，默认为"relu"
        """
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model  # 前馈网络维度
        self.self_attention = self_attention  # 自注意力层
        self.cross_attention = cross_attention  # 交叉注意力层
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)  # 第一个卷积层
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)  # 第二个卷积层
        self.norm1 = nn.LayerNorm(d_model)  # 第一个归一化层
        self.norm2 = nn.LayerNorm(d_model)  # 第二个归一化层
        self.norm3 = nn.LayerNorm(d_model)  # 第三个归一化层
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.activation = F.relu if activation == "relu" else F.gelu  # 激活函数

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        前向传播函数

        Args:
            x (Tensor): 输入张量
            cross (Tensor): 交叉注意力输入张量
            x_mask (Tensor, optional): 输入掩码，默认为None
            cross_mask (Tensor, optional): 交叉注意力掩码，默认为None
            tau (Tensor, optional): 时间衰减因子，默认为None
            delta (Tensor, optional): 时间间隔，默认为None

        Returns:
            Tensor: 编码后的张量
        """
        B, L, D = cross.shape  # 获取交叉注意力输入的形状
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)  # 提取全局token
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))  # 重塑全局token形状
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn  # 残差连接
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)  # 将全局token与序列连接

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  # 第一个卷积层和激活函数
        y = self.dropout(self.conv2(y).transpose(-1, 1))  # 第二个卷积层

        return self.norm3(x + y)  # 残差连接和归一化


class TimeXer(nn.Module):
    """TimeXer2模型：专门用于时间序列分类任务的模型

    Args:
        configs: 模型配置参数，包含seq_len, d_model, num_class等
    """

    def __init__(self, configs):
        """
        初始化TimeXer2分类模型

        Args:
            configs (object): 配置对象，包含模型参数
        """
        super(TimeXer, self).__init__()
        self.seq_len: int = configs.seq_len  # 序列长度
        self.num_class = configs.num_class  # 分类数量
        self.d_model = configs.d_model  # 模型维度
        self.patch_len:int = configs.patch_len  # patch长度
        self.patch_num = int(configs.seq_len // configs.patch_len)  # patch数量
        self.n_vars:int = configs.enc_in  # 输入变量数量

        # Embedding
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)  # 编码器嵌入层
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)  # 数据嵌入层

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # 分类任务专用投影层
        self.act = F.gelu  # 激活函数
        self.dropout = nn.Dropout(configs.dropout)  # Dropout层
        self.projection = nn.Linear(configs.d_model * (self.patch_num + 1) * self.n_vars, configs.num_class)  # 分类投影层

    def forward(self, x_enc, x_mark_enc=None):
        """前向传播函数：处理输入序列并输出分类结果

        Args:
            x_enc (Tensor): 输入序列数据，形状为 [B, T, C]
            x_mark_enc (Tensor, optional): 时间标记数据，用于掩码填充部分，形状为 [B, T]，默认为None

        Returns:
            output (Tensor): 分类结果，形状为 [B, num_class]
        """
        # 数据嵌入
        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))  # 编码器嵌入
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)  # 数据嵌入

        # 通过编码器
        enc_out = self.encoder(en_embed, ex_embed)  # 编码器输出
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)  # 调整输出维度顺序

        # 分类头处理
        output = self.act(enc_out)  # 激活函数
        output = self.dropout(output)  # Dropout

        # 展平特征并投影到类别空间
        output = output.reshape(output.shape[0], -1)  # [B, (patch_num + 1) * d_model * n_vars]，展平特征
        output = self.projection(output)  # [B, num_class]，分类投影
        return output