# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    功能：该类使用离散傅里叶变换（DFT）对输入的时间序列进行分解，将其分解为趋势项（x_trend）和季节性项（x_season）。
    参数：
    top_k: 一个整数，默认值为5，表示在傅里叶变换中保留的最高频成分的数量，用于重构季节性项
    """

    def __init__(self, top_k: int = 5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        """
        输入参数：
        x: 输入的时间序列数据。
        返回值：
        x_season: 提取出来的季节性成分。
        x_trend: 提取出来的趋势成分。

        """
        #对输入数据进行实值快速傅里叶变换（RFFT），将其转换到频域。
        xf = torch.fft.rfft(x)
        # 计算傅里叶系数的绝对值，即各个频率的幅值。
        freq = abs(xf)
        #去除直流分量（即均值部分），只关注变化部分。
        freq[0] = 0
        # 找出幅值最大的前 top_k 个频率及其索引。
        top_k_freq, top_list = torch.topk(freq, k=self.top_k)
        #将小于或等于最小前top_k幅值的频率成分设置为0，起到滤波的作用。
        xf[freq <= top_k_freq.min()] = 0
        #对过滤后的频域数据进行逆实值快速傅里叶变换（IRFFT），将其转换回时域，得到季节性项。
        x_season = torch.fft.irfft(xf)
        #通过从原始数据中减去季节性项，得到趋势项。
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern

    多尺度季节特征混合模块

        功能：实现自顶向下的多尺度季节特征混合，通过下采样层将高频季节特征逐步与低频特征融合

        参数：
        configs - 配置对象，包含以下属性：
            seq_len: 序列长度
            down_sampling_window: 下采样窗口大小
            down_sampling_layers: 下采样层数
    """

    def __init__(self, configs):
        """
        初始化多尺度季节特征混合模块

        构建下采样层列表，每层包含：
        1. 线性变换层(降低维度)
        2. GELU激活函数
        3. 线性变换层(保持维度)
        """

        super(MultiScaleSeasonMixing, self).__init__()
        # 构建多尺度下采样层
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        """
        前向传播过程

        参数：
        season_list - 包含不同尺度季节特征的列表，按从高频到低频排序

        返回：
        out_season_list - 混合后的多尺度季节特征列表
        """

        # mixing high->low
        out_high = season_list[0] # 最高频特征
        out_low = season_list[1]  # 次高频特征
        out_season_list = [out_high.permute(0, 2, 1)]  # 转换维度并存储结果
        # 逐层进行特征混合
        for i in range(len(season_list) - 1):
            # 高频特征下采样并与低频特征融合
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res  # 特征融合
            # 更新高频特征为当前融合结果
            out_high = out_low
            # 如果还有更低频的特征，准备下一轮融合
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
     多尺度趋势特征混合模块

    功能：实现自底向上的多尺度趋势特征混合，通过上采样层将低频趋势特征逐步与高频特征融合

    参数：
    configs - 配置对象，包含以下属性：
        seq_len: 序列长度
        down_sampling_window: 下采样窗口大小
        down_sampling_layers: 下采样层数(决定上采样层数)
    """

    def __init__(self, configs):
        """
        初始化多尺度趋势特征混合模块

        构建上采样层列表，每层包含：
        1. 线性变换层(提升维度)
        2. GELU激活函数
        3. 线性变换层(保持维度)

        注意：层顺序与下采样层相反
        """
        super(MultiScaleTrendMixing, self).__init__()
        #构建多尺度上采样层(按从低频到高频的顺序)
        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):
        """
        前向传播过程

        参数：
        trend_list - 包含不同尺度趋势特征的列表，按从高频到低频排序

        返回：
        out_trend_list - 混合后的多尺度趋势特征列表
        """
        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    """
    历史数据可分解混合模块

    功能：将输入序列分解为季节性和趋势性成分，分别进行多尺度特征混合后重新组合

    参数：
    configs - 配置对象，包含以下属性：
        seq_len: 序列长度
        pred_len: 预测长度
        down_sampling_window: 下采样窗口大小
        d_model: 模型维度
        dropout: dropout概率
        channel_independence: 是否通道独立
        decomp_method: 分解方法('moving_avg'或'dft_decomp')
        moving_avg: 移动平均窗口大小(当decomp_method为'moving_avg'时)
        top_k: 保留的傅里叶分量数(当decomp_method为'dft_decomp'时)
        d_ff: 前馈网络维度
    """
    def __init__(self, configs):
        """
        初始化历史数据可分解混合模块

        主要组件：
        1. 序列分解模块(移动平均或傅里叶分解)
        2. 季节特征混合模块(MultiScaleSeasonMixing)
        3. 趋势特征混合模块(MultiScaleTrendMixing)
        4. 特征交叉处理层(当channel_independence=False时)
        """
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        #self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        # 基础网络层
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence
        # 序列分解模块选择
        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')
        # 通道交叉处理层(仅在非通道独立模式下使用)
        if not configs.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season 多尺度特征混合模块
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        #输出处理层
        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        """
        前向传播过程

        参数：
        x_list - 包含不同尺度输入特征的列表

        返回：
        out_list - 处理后的多尺度特征列表
        """
        # 记录各尺度序列长度
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        # 序列分解阶段
        season_list = []
        trend_list = []
        for x in x_list:
            # 分解为季节性和趋势性成分
            season, trend = self.decompsition(x)
            # 非通道独立模式下进行特征交叉处理
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            # 调整维度并存储
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # # 多尺度特征混合阶段
        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)  # 季节性成分混合
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list) # 趋势性成分混合

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            # 合并季节性和趋势性成分
            out = out_season + out_trend
            # 通道独立模式下的残差连接
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class TimeMixer(nn.Module):
    """
    时间序列混合模型(TimeMixer) - 分类任务专用版本

    功能：针对分类任务优化的多尺度时间序列特征提取和分类模型
    主要特点：
    1. 支持多种下采样方法(max/avg/conv)
    2. 多尺度特征处理
    3. 通道独立/非独立模式

    参数：
    configs - 配置对象，包含以下关键属性：
        task_name = 'classification'
        features = 'M' # [M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate'
        seq_len = seq_len  # 输入序列长度
        use_norm = True
        enc_in = enc_in  # 输入特征维度(特征数)
        d_model = 128  # 模型隐藏层维度 512
        e_layers = 2  # 编码器层数 2

        #多尺度参数
        down_sampling_window = 4 # 下采样窗口大小
        down_sampling_layers = 2  # 下采样层数(通过normalize_layers长度推断)
        down_sampling_method = 'avg'  #下采样方法(max/avg/conv)

        #正则化参数
        dropout = 0.1  # dropout
        use_norm = True # 是否使用归一化

        channel_independence = False # 是否使用通道独立

        embed = 'timeF'  # 嵌入类型 timeF, fixed, learned
        freq = 'd'        # 时间频率 [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]

        moving_avg = 25  # 移动平均窗口大小 #25
        decomp_method = 'dft_decomp'  # 分解方法(moving_avg/dft_decomp)
        top_k = 5  # DFT分解保留的top_k频率 5

        d_ff = 256 # FFN隐藏层维度

        num_class = 2
    """
    def __init__(self, configs):
        super(TimeMixer, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
       # self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in

        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def __multi_scale_process_inputs(self, x_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return [x_enc]

        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

        return x_enc_sampling_list

    def classification(self, x_enc):
        x_enc = self.__multi_scale_process_inputs(x_enc)
        x_list = x_enc

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        enc_out = enc_out_list[0]
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output


    def forward(self, x_enc):
        dec_out = self.classification(x_enc)
        return dec_out  # [B, N]






