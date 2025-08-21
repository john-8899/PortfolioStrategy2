# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegRNN(nn.Module):
    """
    SegRNN: 专门用于时间序列分类任务的模型
    """

    def __init__(self, configs):
        super(SegRNN, self).__init__()

        # get parameters
        self.seq_len: int = configs.seq_len
        self.enc_in: int = configs.enc_in
        self.d_model: int = configs.d_model
        self.dropout = configs.dropout
        self.num_class: int = configs.num_class

        self.seg_len: int= configs.seg_len

        self.seg_num_x: int = self.seq_len  // self.seg_len
        self.seg_num_y: int = self.seq_len // self.seg_len
        #print(f"seg_num_x: {self.seg_num_x},seg_num_y: {self.seg_num_y}")

        # building model
        # 值嵌入层：将分割后的序列片段映射到高维特征空间
        # 输入维度为seg_len，输出维度为d_model
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.ReLU()
        )
        # GRU循环神经网络层：捕捉序列的时间依赖关系
        # 输入大小和隐藏层大小均为d_model，使用单层GRU，按批次优先处理
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                          batch_first=True, bidirectional=False)
        # 位置嵌入参数：学习序列段的位置信息
        # 维度为[seg_num_x, d_model//2]，seg_num_x为序列段数量
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        # 通道嵌入参数：学习不同输入通道的特征表示
        # 维度为[enc_in, d_model//2]，enc_in为输入通道数
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seq_len)
        )

        # 分类任务专用投影层
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(
            configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        # b:batch_size c:channel_size s:seq_len
        # d:d_model w:seg_len n:seg_num_x
        batch_size = x.size(0)
        #print(f"x 0形状：{x.size()}")
        # normalization and permute     b,s,c -> b,c,s
        seq_last = x[:, -1:, :].detach()
        #print(f"seq_last形状：{seq_last.size()}")
        x = (x - seq_last).permute(0, 2, 1)  # b,c,s
        #print(f"x 1形状：{x.size()}")
        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        #x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))
        x = self.valueEmbedding(x.reshape(-1, 1, self.seq_len))
        #print(f"x 2形状：{x.size()}")

        # encoding
        _, hn = self.rnn(x)  # bc,n,d  1,bc,d

        # n,d//2 -> 1,n,d//2 -> c,n,d//2
        # c,d//2 -> c,1,d//2 -> c,n,d//2
        # c,n,d -> cm,1,d -> bcm, 1, d
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)

        #修改前
        #_, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))  # bcm,1,d  1,bcm,d

        #print(f"pos_emb形状：{pos_emb.size()}")
        # 直接使用原始SegRNN方法，但确保维度正确
        hidden_states = hn.repeat(1, 1, self.seg_num_y)
        #print(f"hn 形状：{hn.size()}")
        #print(f"hidden_states形状：{hidden_states.size()}")
        hidden_states= hidden_states.view(1, -1, self.d_model)
        #print(f"hidden_states view形状：{hidden_states.size()}")

        _, hy = self.rnn(pos_emb, hidden_states)  # bcm,1,d  1,bcm,d
        #print(f"hy 形状：{hy.size()}")

        # 1,bcm,d -> 1,bcm,w -> b,c,s
        y = self.predict(hy).view(batch_size, self.seg_num_x, self.enc_in, self.seq_len).mean(dim=1)
        #print(f"y 形状：{y.size()}")

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_last
        return y

    def forward(self, x_enc, x_mark_enc=None):
        """
        前向传播函数：处理输入序列并输出分类结果

        Args:
            x_enc: 输入序列数据，形状为 [B, T, C]
            x_mark_enc: 时间标记数据（可选），用于掩码填充部分，形状为 [B, T]

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        # Encoder
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