# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from layers.Embed import PositionalEmbedding
from layers.StandardNorm import Normalize
from layers.TimeFilter_layers import TimeFilter_Backbone


class PatchEmbed(nn.Module):
    """Patch嵌入模块：将时间序列分割为patch并进行线性投影

    Args:
        dim: 嵌入维度
        patch_len: patch长度
        stride: 滑动步长，默认为patch_len
        pos: 是否使用位置编码
    """

    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len: int = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = pos
        if self.pos:
            pos_emb_theta = 10000
            self.pe = PositionalEmbedding(dim, pos_emb_theta)

    def forward(self, x):
        # x: [B, N*L] - 展平的时序数据
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [B, N*L/patch_len, patch_len] 分割为patch
        x = self.patch_proj(x)  # [B, N*L/patch_len, dim] 线性投影
        if self.pos:
            x += self.pe(x)
        return x


class TimeFilter(nn.Module):
    """TimeFilter模型：专门用于时间序列分类任务的模型
    Args:
        configs: 模型配置参数，包含seq_len, d_model, num_class等
        dim
        含义：表示模型的隐藏层维度（embedding dimension），即每个patch经过线性变换后的特征维度。
        合理范围：通常设置为2的幂次，例如64、128、256、512等。具体值取决于模型复杂度需求
        d_ff
        含义：前馈神经网络（Feed Forward Network）中的中间层维度。
        合理范围：通常是dim的2~4倍，常见取值为dim * 4。
         patch_len
        含义：时间序列被切分成的小段（patch）的长度。
        合理范围：一般根据时间序列特性和任务需求设定，常见值包括8、16、32、64等。

        stride
        含义：滑动窗口在时间序列上移动的步长。
        合理范围：通常等于patch_len（无重叠）或小于patch_len（有重叠）。若大于patch_len会导致信息丢失
        n_vars
        含义：输入时间序列的变量数量（通道数），对应于输入数据的特征维度。
        合理范围：由实际应用场景决定，等于输入数据的特征数（如股票数量、传感器数量等）。
        num_patches
        含义：时间序列被分割成的patch总数。
         alpha
        含义：控制邻接矩阵稀疏性的阈值比例，在mask_topk函数中用于保留一定比例的最大权重连接。
        合理范围：[0.0, 1.0]之间，常用值为0.1、0.2、0.5等。
        top_p
        含义：在MoE（Mixture of Experts）门控机制中使用的概率累积阈值，用于选择最重要的专家组合。
        合理范围：[0.0, 1.0]之间，常用值为0.5、0.7、0.9等
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.num_class = configs.num_class
        self.dim : int = configs.d_model
        self.d_ff = configs.d_ff
        self.patch_len = configs.patch_len
        self.stride = self.patch_len
        self.n_vars = configs.enc_in
        # 计算patches数量，专门针对分类任务优化
        self.num_patches = int((self.seq_len * configs.enc_in - self.patch_len) / self.stride + 1)

        # Filter参数
        self.alpha = 0.1 if configs.alpha is None else configs.alpha
        self.top_p = 0.5 if configs.top_p is None else configs.top_p

        # 嵌入层
        self.patch_embed = PatchEmbed(self.dim, self.patch_len, self.stride, configs.pos)

        # TimeFilter骨干网络
        self.backbone = TimeFilter_Backbone(
            self.dim, self.n_vars, self.d_ff,
            configs.n_heads, configs.e_layers, self.top_p, configs.dropout,
            self.seq_len * self.n_vars // self.patch_len
        )

        # 分类任务专用头
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(self.dim * self.num_patches, configs.num_class)

        # 标准化层
        self.use_RevIN = False
        self.norm = Normalize(configs.enc_in, affine=self.use_RevIN)

    def _get_mask(self, device):
        """生成掩码用于TimeFilter骨干网络

        Args:
            device: 设备类型

        Returns:
            masks: 掩码张量，用于TimeFilter的自注意力计算
        """
        dtype = torch.float32
        L = self.configs.seq_len * self.configs.enc_in // self.configs.patch_len
        N = self.configs.seq_len // self.configs.patch_len
        masks = []
        for k in range(L):
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(device)
            T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N) & (torch.arange(L) != k)).to(
                dtype).to(device)
            ST = torch.ones(L).to(dtype).to(device) - S - T
            ST[k] = 0.0
            masks.append(torch.stack([S, T, ST], dim=0))
        masks = torch.stack(masks, dim=0)
        return masks

    def forward(self, x_enc, x_mark_enc=None):
        """前向传播函数：处理输入序列并输出分类结果

        Args:
            x_enc: 输入序列数据，形状为 [B, T, C]
            x_mark_enc: 时间标记数据，用于掩码填充部分，形状为 [B, T]

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        # x: [B, T, C]
        B, T, C = x_enc.shape

        # 数据标准化
        x = self.norm(x_enc, 'norm')

        # 数据重塑和patch嵌入
        # x: [B, C, T] -> [B, C*T] 展平
        x = x.permute(0, 2, 1).reshape(-1, C * T)
        # x: [B, N, D]  N = [C*T / patch_len]
        x = self.patch_embed(x)

        # 通过TimeFilter骨干网络
        x, _ = self.backbone(x, self._get_mask(x.device), self.alpha)

        # 分类头处理
        output = self.dropout(x.flatten(start_dim=1))  # 展平特征
        output = self.projection(output)  # 投影到类别空间 [B, num_class]

        return output