# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from torch import Tensor
from typing import Optional


def get_known_len(embed_type, freq):
    """获取已知特征的长度

    Args:
        embed_type: 嵌入类型
        freq: 时间频率

    Returns:
        int: 已知特征的维度
    """
    if embed_type != 'timeF':
        if freq == 't':
            return 5
        else:
            return 4
    else:
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        # 添加默认值以防freq不在映射中
        return freq_map.get(freq,4) # 返回默认值


class TFTTemporalEmbedding(nn.Module):
    """TFT时间嵌入模块

    Args:
        d_model: 模型维度
        embed_type: 嵌入类型
        freq: 时间频率
    """

    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TFTTemporalEmbedding, self).__init__()
        self.embed_type = embed_type
        self.freq = freq

        # 根据频率创建不同的嵌入层
        if embed_type == 'fixed':
            if freq == 't':
                self.minute_embed = nn.Embedding(60, d_model)
            self.hour_embed = nn.Embedding(24, d_model)
            self.weekday_embed = nn.Embedding(7, d_model)
            self.day_embed = nn.Embedding(32, d_model)
            self.month_embed = nn.Embedding(13, d_model)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入时间标记，形状为 [B, T, D]

        Returns:
            Tensor: 时间嵌入，形状为 [B, T, C, d_model]
        """
        x = x.long()
        if self.embed_type == 'fixed':
            minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
            hour_x = self.hour_embed(x[:, :, 3])
            weekday_x = self.weekday_embed(x[:, :, 2])
            day_x = self.day_embed(x[:, :, 1])
            month_x = self.month_embed(x[:, :, 0])

            embedding_x = torch.stack([month_x, day_x, weekday_x, hour_x, minute_x], dim=-2) if hasattr(
                self, 'minute_embed') else torch.stack([month_x, day_x, weekday_x, hour_x], dim=-2)
            return embedding_x
        else:
            return None


class TFTTimeFeatureEmbedding(nn.Module):
    """TFT时间特征嵌入模块

    Args:
        d_model: 模型维度
        embed_type: 嵌入类型
        freq: 时间频率
    """

    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TFTTimeFeatureEmbedding, self).__init__()
        d_inp = get_known_len(embed_type, freq)
        self.embed = nn.ModuleList([nn.Linear(1, d_model, bias=False) for _ in range(d_inp)])

    def forward(self, x):
        """前向传播

        Args:
            x: 输入时间特征，形状为 [B, T, D]

        Returns:
            Tensor: 时间特征嵌入，形状为 [B, T, C, d_model]
        """
        return torch.stack([embed(x[:, :, i].unsqueeze(-1)) for i, embed in enumerate(self.embed)], dim=-2)



class TFTEmbedding(nn.Module):
    """TFT嵌入模块，整合静态、观测和已知特征

    Args:
        configs: 配置参数
    """

    def __init__(self, configs):
        super(TFTEmbedding, self).__init__()
        self.seq_len = configs.seq_len
        self.static_len = getattr(configs, 'static_len', 0)
        self.observed_len = getattr(configs, 'observed_len', configs.enc_in)

        # 静态特征嵌入
        self.static_embedding = nn.ModuleList([DataEmbedding(1, configs.d_model, dropout=configs.dropout)
                                               for _ in range(self.static_len)]) if self.static_len else None
        # 观测特征嵌入
        self.observed_embedding = nn.ModuleList([DataEmbedding(1, configs.d_model, dropout=configs.dropout)
                                                 for _ in range(self.observed_len)])
        # 已知特征嵌入
        self.known_embedding = TFTTemporalEmbedding(configs.d_model, configs.embed, configs.freq) \
            if configs.embed != 'timeF' else TFTTimeFeatureEmbedding(configs.d_model, configs.embed, configs.freq)

    def forward(self, x_enc, x_mark_enc):
        """前向传播

        Args:
            x_enc: 输入序列数据，形状为 [B, T, C]
            x_mark_enc: 时间标记数据，形状为 [B, T, D]

        Returns:
            tuple: (static_input, observed_input, known_input)
        """
        if self.static_len > 0:
            # static_input: [B, C, d_model]
            static_input = torch.stack([embed(x_enc[:, :1, i].unsqueeze(-1), None).squeeze(1)
                                        for i, embed in enumerate(self.static_embedding)], dim=-2)
        else:
            static_input = None

        # observed_input: [B, T, C, d_model]
        observed_input = torch.stack([embed(x_enc[:, :, i].unsqueeze(-1), None)
                                      for i, embed in enumerate(self.observed_embedding)], dim=-2)

        # known_input: [B, T, C, d_model]
        known_input = self.known_embedding(x_enc)

        return static_input, observed_input, known_input


class GLU(nn.Module):
    """门控线性单元

    Args:
        input_size: 输入维度
        output_size: 输出维度
    """

    def __init__(self, input_size, output_size):
        super(GLU, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.glu = nn.GLU()

    def forward(self, x):
        """前向传播

        Args:
            x: 输入张量

        Returns:
            Tensor: 输出张量
        """
        a = self.fc1(x)
        b = self.fc2(x)
        return self.glu(torch.cat([a, b], dim=-1))


class GateAddNorm(nn.Module):
    """门控加归一化模块

    Args:
        input_size: 输入维度
        output_size: 输出维度
    """

    def __init__(self, input_size, output_size):
        super(GateAddNorm, self).__init__()
        self.glu = GLU(input_size, input_size)
        self.projection = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x, skip_a):
        """前向传播

        Args:
            x: 输入张量
            skip_a: 跳跃连接张量

        Returns:
            Tensor: 输出张量
        """
        x = self.glu(x)
        x = x + skip_a
        return self.layer_norm(self.projection(x))


class GRN(nn.Module):
    """门控残差网络

    Args:
        input_size: 输入维度
        output_size: 输出维度
        hidden_size: 隐藏层维度
        context_size: 上下文维度
        dropout: Dropout比率
    """

    def __init__(self, input_size, output_size, hidden_size=None, context_size=None, dropout=0.0):
        super(GRN, self).__init__()
        hidden_size = input_size if hidden_size is None else hidden_size
        self.lin_a = nn.Linear(input_size, hidden_size)
        self.lin_c = nn.Linear(context_size, hidden_size) if context_size is not None else None
        self.lin_i = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.project_a = nn.Linear(input_size, hidden_size) if hidden_size != input_size else nn.Identity()
        self.gate = GateAddNorm(hidden_size, output_size)

    def forward(self, a: Tensor, c: Optional[Tensor] = None):
        """前向传播

        Args:
            a: 输入张量，形状为 [B, T, d]
            c: 上下文张量，形状为 [B, d]

        Returns:
            Tensor: 输出张量
        """
        # a: [B,T,d], c: [B,d]
        x = self.lin_a(a)
        if c is not None:
            x = x + self.lin_c(c).unsqueeze(1)
        x = F.elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)
        return self.gate(x, self.project_a(a))


class VariableSelectionNetwork(nn.Module):
    """变量选择网络

    Args:
        d_model: 模型维度
        variable_num: 变量数量
        dropout: Dropout比率
    """

    def __init__(self, d_model, variable_num, dropout=0.0):
        super(VariableSelectionNetwork, self).__init__()
        self.joint_grn = GRN(d_model * variable_num, variable_num, hidden_size=d_model, context_size=d_model,
                             dropout=dropout)
        self.variable_grns = nn.ModuleList([GRN(d_model, d_model, dropout=dropout) for _ in range(variable_num)])

    def forward(self, x: Tensor, context: Optional[Tensor] = None):
        """前向传播

        Args:
            x: 输入张量，形状为 [B, T, C, d] 或 [B, C, d]
            context: 上下文张量，形状为 [B, d]

        Returns:
            Tensor: 选择后的张量，形状为 [B, T, d] 或 [B, d]
        """
        # x: [B,T,C,d] or [B,C,d]
        # selection_weights: [B,T,C] or [B,C]
        # x_processed: [B,T,d,C] or [B,d,C]
        # selection_result: [B,T,d] or [B,d]
        x_flattened = torch.flatten(x, start_dim=-2)
        selection_weights = self.joint_grn(x_flattened, context)
        selection_weights = F.softmax(selection_weights, dim=-1)

        x_processed = torch.stack([grn(x[..., i, :]) for i, grn in enumerate(self.variable_grns)], dim=-1)

        selection_result = torch.matmul(x_processed, selection_weights.unsqueeze(-1)).squeeze(-1)
        return selection_result


class StaticCovariateEncoder(nn.Module):
    """静态协变量编码器

    Args:
        d_model: 模型维度
        static_len: 静态变量数量
        dropout: Dropout比率
    """

    def __init__(self, d_model, static_len, dropout=0.0):
        super(StaticCovariateEncoder, self).__init__()
        self.static_vsn = VariableSelectionNetwork(d_model, static_len) if static_len else None
        self.grns = nn.ModuleList([GRN(d_model, d_model, dropout=dropout) for _ in range(4)])

    def forward(self, static_input):
        """前向传播

        Args:
            static_input: 静态输入张量，形状为 [B, C, d]

        Returns:
            list: 编码后的上下文张量列表
        """
        # static_input: [B,C,d]
        if static_input is not None:
            static_features = self.static_vsn(static_input)
            return [grn(static_features) for grn in self.grns]
        else:
            return [None] * 4


class InterpretableMultiHeadAttention(nn.Module):
    """可解释的多头注意力机制

    Args:
        configs: 配置参数
    """

    def __init__(self, configs):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.n_heads:int = configs.n_heads
        assert configs.d_model % configs.n_heads == 0
        self.d_head:int = configs.d_model // configs.n_heads
        self.qkv_linears = nn.Linear(configs.d_model, (2 * self.n_heads + 1) * self.d_head, bias=False)
        self.out_projection = nn.Linear(self.d_head, configs.d_model, bias=False)
        self.out_dropout = nn.Dropout(configs.dropout)
        self.scale = self.d_head ** -0.5
        self.register_buffer("mask", torch.triu(torch.full((configs.seq_len, configs.seq_len), float('-inf')), 1))

    def forward(self, x):
        """前向传播

        Args:
            x: 输入张量，形状为 [B, T, d_model]

        Returns:
            Tensor: 注意力输出，形状为 [B, T, d_model]
        """
        # Q,K,V are all from x
        B, T, d_model = x.shape
        qkv = self.qkv_linears(x)
        q, k, v = qkv.split((self.n_heads * self.d_head, self.n_heads * self.d_head, self.d_head), dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_heads, self.d_head)
        v = v.view(B, T, self.d_head)

        attention_score = torch.matmul(q.permute((0, 2, 1, 3)), k.permute((0, 2, 3, 1)))  # [B,n,T,T]
        attention_score.mul_(self.scale)
        attention_score = attention_score + self.mask
        attention_prob = F.softmax(attention_score, dim=3)  # [B,n,T,T]

        attention_out = torch.matmul(attention_prob, v.unsqueeze(1))  # [B,n,T,d]
        attention_out = torch.mean(attention_out, dim=1)  # [B,T,d]
        out = self.out_projection(attention_out)
        out = self.out_dropout(out)  # [B,T,d]
        return out


class TemporalFusionDecoder(nn.Module):
    """时序融合解码器

    Args:
        configs: 配置参数
    """

    def __init__(self, configs):
        super(TemporalFusionDecoder, self).__init__()
        self.seq_len = configs.seq_len

        self.history_encoder = nn.LSTM(configs.d_model, configs.d_model, batch_first=True)
        self.gate_after_lstm = GateAddNorm(configs.d_model, configs.d_model)
        self.enrichment_grn = GRN(configs.d_model, configs.d_model, context_size=configs.d_model,
                                  dropout=configs.dropout)
        self.attention = InterpretableMultiHeadAttention(configs)
        self.gate_after_attention = GateAddNorm(configs.d_model, configs.d_model)
        self.position_wise_grn = GRN(configs.d_model, configs.d_model, dropout=configs.dropout)
        self.gate_final = GateAddNorm(configs.d_model, configs.d_model)

    def forward(self, history_input, c_c, c_h, c_e):
        """前向传播

        Args:
            history_input: 历史输入，形状为 [B, T, d]
            c_c: 上下文向量c
            c_h: 上下文向量h
            c_e: 上下文向量e

        Returns:
            Tensor: 解码器输出，形状为 [B, T, d]
        """
        # history_input: [B,T,d]
        # c_c, c_h, c_e: [B,d]
        # LSTM
        c = (c_c.unsqueeze(0), c_h.unsqueeze(0)) if c_c is not None and c_h is not None else None
        temporal_features, _ = self.history_encoder(history_input, c)

        # Skip connection
        temporal_features = self.gate_after_lstm(temporal_features, history_input)  # [B,T,d]

        # Static enrichment
        enriched_features = self.enrichment_grn(temporal_features, c_e)  # [B,T,d]

        # Temporal self-attention
        attention_out = self.attention(enriched_features)  # [B,T,d]

        # Gate after attention
        attention_out = self.gate_after_attention(attention_out, enriched_features)

        # Position-wise feed-forward
        out = self.position_wise_grn(attention_out)  # [B,T,d]

        # Final skip connection
        out = self.gate_final(out, temporal_features)
        return out


class TemporalFusionTransformer(nn.Module):
    """TemporalFusionTransformer模型：专门用于时间序列分类任务的模型

    Args:
        configs: 模型配置参数，包含seq_len, d_model, num_class等
    """

    def __init__(self, configs):
        super(TemporalFusionTransformer, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.num_class = configs.num_class
        self.d_model = configs.d_model

        # 特征维度
        self.static_len = getattr(configs, 'static_len', 0)
        self.observed_len = getattr(configs, 'observed_len', configs.enc_in)
        self.known_len = get_known_len(configs.embed, configs.freq)

        # 嵌入层
        self.embedding = TFTEmbedding(configs)
        # 静态编码器
        self.static_encoder = StaticCovariateEncoder(configs.d_model, self.static_len)
        # 变量选择网络
        self.history_vsn = VariableSelectionNetwork(configs.d_model, self.observed_len + self.known_len)
        # 时序融合解码器
        self.temporal_fusion_decoder = TemporalFusionDecoder(configs)
        # 分类头
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forward(self, x_enc, x_mark_enc=None):
        """前向传播函数：处理输入序列并输出分类结果

        Args:
            x_enc: 输入序列数据，形状为 [B, T, C]
            x_mark_enc: 时间标记数据，用于掩码填充部分，形状为 [B, T, D]

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        # 数据嵌入
        # static_input: [B,C,d], observed_input:[B,T,C,d], known_input: [B,T,C,d]
        static_input, observed_input, known_input = self.embedding(x_enc, x_mark_enc)

        # 静态上下文
        # c_s,...,c_e: [B,d]
        c_s, c_c, c_h, c_e = self.static_encoder(static_input)

        # 时序输入选择
        history_input = torch.cat([observed_input, known_input], dim=-2)
        history_input = self.history_vsn(history_input, c_s)

        # TFT主流程
        # history_input: [B,T,d]
        temporal_features = self.temporal_fusion_decoder(history_input, c_c, c_h, c_e)

        # 分类头处理
        output = self.act(temporal_features)
        output = self.dropout(output)
        # 展平特征并投影到类别空间
        output = output.reshape(output.shape[0], -1)  # [B, seq_len * d_model]
        output = self.projection(output)  # [B, num_class]
        return output