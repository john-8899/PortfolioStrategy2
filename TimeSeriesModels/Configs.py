# -*- coding:utf-8 -*-

class TimeMixer_Configs:
    def __init__(self,seq_len,enc_in):
        self.task_name = 'classification'
        self.features = 'MS' # [M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate'
                                          #M：多变量预测多变量，S：单变量预测单变量，MS：多变量预测单变量
        self.seq_len = seq_len  # 输入序列长度
        self.use_norm = True
        self.enc_in = enc_in  # 输入特征维度(特征数)
        self.d_model = 128  # 模型隐藏层维度 512
        self.e_layers = 3  # 编码器层数 2

        #多尺度参数
        self.down_sampling_window = 5 # 下采样窗口大小
        self.down_sampling_layers = 2  # 下采样层数(通过normalize_layers长度推断)
        self.down_sampling_method = 'max'  #下采样方法(max/avg/conv)

        #正则化参数
        self.dropout = 0.1  # dropout
        self.use_norm = True # 是否使用归一化

        self.channel_independence = False # 是否使用通道独立

        self.embed = 'timeF'  # 嵌入类型 timeF, fixed, learned
        self.freq = 'd'        # 时间频率 [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]

        self.moving_avg = 10  # 移动平均窗口大小 #25
        self.decomp_method = 'dft_decomp'  # 分解方法(moving_avg/dft_decomp)
        self.top_k = 5  # DFT分解保留的top_k频率 5

        self.d_ff = 256 # FFN隐藏层维度

        self.num_class = 2

class TimesNet_configs:
    def __init__(self,seq_len,enc_in):
        """

        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        #TimesBlock模块：使用FFT提取周期特征并通过2D卷积处理 卷积层参数(in_channels, out_channels, num_kernels)
        self.d_model = 128 # 模型隐藏层维度 512 d_model=in_channels
        self.d_ff = 256 # d_ff = (out_channels)
        self.num_kernels = 3 #num_kernels
        self.top_k = 3 # 快速傅里叶变换(FFT)

        self.e_layers = 3 # 编码器层数

        #嵌入层参数：嵌入类型 timeF, fixed, learned
        """
            timeF：时间序列预测、时间特征编码
            fixed：NLP中的位置编码、图结构编码
            learned：文本分类、机器翻译、推荐系统等
        """
        self.embed = 'timeF'  # 嵌入类型 timeF, fixed, learned
        self.freq = 'd'  # 时间频率 [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]

        self.dropout = 0.1 # dropout
        self.num_class = 2  # 分类数

class TimesNet_configs2:
    def __init__(self,seq_len,enc_in,d_model,d_ff,num_kernels,top_k,e_layers,freq,dropout,num_class=2):
        """

        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        #TimesBlock模块：使用FFT提取周期特征并通过2D卷积处理 卷积层参数(in_channels, out_channels, num_kernels)
        self.d_model = d_model # 模型隐藏层维度 512 d_model=in_channels
        self.d_ff = d_ff # d_ff = (out_channels)
        self.num_kernels = num_kernels #num_kernels
        self.top_k = top_k # 快速傅里叶变换(FFT)

        self.e_layers = e_layers # 编码器层数

        #嵌入层参数：嵌入类型 timeF, fixed, learned
        """
            timeF：时间序列预测、时间特征编码
            fixed：NLP中的位置编码、图结构编码
            learned：文本分类、机器翻译、推荐系统等
        """
        self.embed = 'timeF'  # 嵌入类型 timeF, fixed, learned
        self.freq = freq  # 时间频率 [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]

        self.dropout = dropout # dropout
        self.num_class = 2  # 分类数

class Nonstationary_configs:
    def __init__(self,seq_len,enc_in):
        """

        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.d_model = 512 # 模型隐藏层维度 512 d_model=in_channels
        self.d_ff = 256 # d_ff = (out_channels)
        self.num_kernels = 3 #num_kernels

        self.e_layers = 2 # 编码器层数

        self.factor = 5 # 注意力机制参数
        self.n_heads = 8
        self.activation = 'relu'
        self.p_hidden_dims = [128,128] # MLP隐藏层维度
        self.p_hidden_layers = 2  # MLP隐藏层数(和上面一个参数配合使用)
        #嵌入层参数：嵌入类型 timeF, fixed, learned
        """
            timeF：时间序列预测、时间特征编码
            fixed：NLP中的位置编码、图结构编码
            learned：文本分类、机器翻译、推荐系统等
        """
        self.embed = 'timeF'  # 嵌入类型 timeF, fixed, learned
        self.freq = 'd'  # 时间频率 [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]

        self.dropout = 0.1 # dropout
        self.num_class = 2  # 分类数

class Informer_configs:
    def __init__(self,seq_len,enc_in):
        """

        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.d_model = 128 # d_model=in_channels
        self.d_ff = 256 # d_ff = (out_channels)

        self.e_layers = 3 # 编码器层数
        self.distil = False #'whether to use distilling in encoder, using this argument means not using distilling

        self.factor = 5 # 注意力机制参数
        self.n_heads = 8
        self.activation = 'relu'

        #嵌入层参数：嵌入类型 timeF, fixed, learned
        """
            timeF：时间序列预测、时间特征编码
            fixed：NLP中的位置编码、图结构编码
            learned：文本分类、机器翻译、推荐系统等
        """
        self.embed = 'timeF'  # 嵌入类型 timeF, fixed, learned
        self.freq = 'd'  # 时间频率 [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]

        self.dropout = 0.1 # dropout
        self.num_class = 2  # 分类数

class Mamba_configs:
    def __init__(self,seq_len,enc_in):
        """

        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.d_model = 64 # d_model=in_channels
        self.d_ff = 128 # d_ff = (out_channels)

        self.expand = 2 #expansion factor for Mamba
        self.d_conv = 3 # conv kernel size for Mamba

        self.e_layers = 2 # 编码器层数

        #嵌入层参数：嵌入类型 timeF, fixed, learned
        """
            timeF：时间序列预测、时间特征编码
            fixed：NLP中的位置编码、图结构编码
            learned：文本分类、机器翻译、推荐系统等
        """
        self.embed = 'timeF'  # 嵌入类型 timeF, fixed, learned
        self.freq = 'd'  # 时间频率 [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]

        self.dropout = 0.1 # dropout
        self.num_class = 2  # 分类数

class MultiPatchFormer_configs:
    def __init__(self,seq_len,enc_in):
        """

        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.d_model = 64 # d_model=in_channels
        self.d_ff = 128 # d_ff = (out_channels)

        self.e_layers = 2 # 编码器层数
        self.n_heads = 8

        self.dropout = 0.1 # dropout
        self.num_class = 2  # 分类数

class TimeXer_configs:
    def __init__(self,seq_len,enc_in):
        """

        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.d_model = 64 # d_model
        self.d_ff = 128 # d_ff
        self.patch_len = 5 # patch_len:16
        self.factor = 5 # 注意力机制参数
        self.activation = 'relu' # gelu/relu

        self.e_layers = 3 # 编码器层数
        self.n_heads = 32 # 注意力头数

        #嵌入层参数：嵌入类型 timeF, fixed, learned
        """
            timeF：时间序列预测、时间特征编码
            fixed：NLP中的位置编码、图结构编码
            learned：文本分类、机器翻译、推荐系统等
        """
        self.embed = 'timeF'  # 嵌入类型 timeF, fixed, learned
        self.freq = 'd'  # 时间频率 [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]

        self.dropout = 0.1 # dropout
        self.num_class = 2  # 分类数

class PatchTST_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        patch_len  通常为8、12、16、24，需要根据seq_len选择合适的值
        stride 通常为patch_len的一半或与patch_len相等
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.d_model = 128 # d_model=in_channels 通常为64、128、256、512，根据数据复杂度和计算资源选择
        self.d_ff = 256 # d_ff = (out_channels) 通常为d_model的2-4倍，如128、256、512

        self.e_layers = 2 # 编码器层数

        self.factor = 5 # 注意力机制参数 通常为3、5、7
        self.n_heads = 16 # 注意力头数 通常为4、8、16、32，需要能被d_model整除
        self.activation = 'relu'# gelu/relu

        self.dropout = 0.1 # dropout
        self.num_class = 2  # 分类数

class iTransformer_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.d_model = 128 # d_model=in_channels 通常为64、128、256、512，根据数据复杂度和计算资源选择
        self.d_ff = 256 # d_ff = (out_channels) 通常为d_model的2-4倍，如128、256、512

        self.e_layers = 2 # 编码器层数

        self.factor = 5 # 注意力机制参数 通常为3、5、7
        self.n_heads = 16 # 注意力头数 通常为4、8、16、32，需要能被d_model整除
        self.activation = 'gelu'# gelu/relu

        #嵌入层参数：嵌入类型 timeF, fixed, learned
        """
            timeF：时间序列预测、时间特征编码
            fixed：NLP中的位置编码、图结构编码
            learned：文本分类、机器翻译、推荐系统等
        """
        self.embed = 'timeF'  # 嵌入类型 timeF, fixed, learned
        self.freq = 'd'  # 时间频率 [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]

        self.dropout = 0.1 # dropout
        self.num_class = 2  # 分类数

class MICN_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)
        self.c_out = enc_in #输出特征维度(特征数)

        self.d_model = 256 # d_model=in_channels 通常为64、128、256、512，根据数据复杂度和计算资源选择

        self.d_layers = 4 # MIC层数

        self.n_heads = 8 # 注意力头数 通常为4、8、16、32，需要能被d_model整除

        #嵌入层参数：嵌入类型 timeF, fixed, learned
        """
            timeF：时间序列预测、时间特征编码
            fixed：NLP中的位置编码、图结构编码
            learned：文本分类、机器翻译、推荐系统等
        """
        self.embed = 'timeF'  # 嵌入类型 timeF, fixed, learned
        self.freq = 'd'  # 时间频率 [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]

        self.dropout = 0.1 # dropout
        self.num_class = 2  # 分类数

class Koopa_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.num_blocks = 5     ## num_blocks: int, number of Koopa blocks Koopa块的数量 默认 3
        self.dynamic_dim = 128  ## dynamic_dim: int, latent dimension of koopman embedding 嵌入的潜在维度 默认为128
        self.hidden_dim = 64    ## hidden_dim: int, hidden dimension of en/decoder  编码器/解码器的隐藏维度 默认为64
        self.hidden_layers = 2  ##hidden_layers: int, number of hidden layers of en/decoder 编码器/解码器的隐藏层数量 默认为2
        self.multistep = False  ## multistep: bool, whether to use approximation for multistep K  否使用多步K的近似 默认： False
        self.alpha = 0.2        ## alpha: float, spectrum filter ratio 光谱滤波比 默认为0.2

        self.dropout = 0.1 # dropout
        self.num_class = 2  # 分类数

class FiLM_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        # 多尺度和窗口大小配置
        # multiscale: 多尺度因子列表，用于扩展输入序列长度
        # window_size: 窗口大小列表，用于HiPPO-LegT变换
        self.multiscale = [1, 2, 4] # 多尺度因子列表 默认[1,2,4]
        self.window_size = [256] # 窗口大小列表 默认[256]
        self.ratio = 0.5 # HiPPO-LegT变换的缩放比例 默认0.5


        self.dropout = 0.1 # dropout
        self.num_class = 2  # 分类数

class SegRNN_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.d_model = 128 # d_model=in_channels 通常为64、128、256、512，根据数据复杂度和计算资源选择
        self.seg_len = 8 # 片段长度 默认16

        self.dropout = 0.1 # dropout
        self.num_class = 2  # 分类数


class TiDE_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.d_model = 128 # hidden_dim 隐藏层维度 / res_hidden 残差块隐藏层维度 默认128
        self.e_layers = 3  # 编码器层数 默认3

        self.bias = True # 是否使用偏置项 默认True
        self.feature_encode_dim = 78 # 特征编码维度 默认2

        self.dropout = 0.05 # dropout
        self.num_class = 2  # 分类数

class SCINet_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.current_level = 4 # 当前递归层级，默认为3
        self.kernel_size = 5 # 卷积核大小 默认5

        self.dropout = 0.05 # dropout
        self.num_class = 2  # 分类数

class LightTS_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.d_model = 128 # 隐藏层维度
        self.chunk_size = 30 # 分块大小 默认为24

        self.dropout = 0.05 # dropout
        self.num_class = 2  # 分类数

class FEDformer_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.d_model = 64 # 作为输入嵌入层（DataEmbedding）的输出维度决定Encoder中各层的特征维度 默认为128
        self.n_heads = 64 # 注意力头数 需要能够整除d_model，以确保每个头的维度是整数 默认8

        self.d_ff = 256 # 前馈网络维度，表示前馈网络中的隐藏层维度 在FEDformer_configs中默认值为256（通常是d_model的2倍）
        self.moving_avg = 25 #时间序列分解中移动平均的窗口大小 用于构建移动平均层，提取时间序列的趋势部分 默认为25
        self.activation = 'gelu' # 激活函数 默认为gelu
        self.e_layers = 3 # 编码器层数 默认为3
        #嵌入层参数：嵌入类型 timeF, fixed, learned
        """
            timeF：时间序列预测、时间特征编码
            fixed：NLP中的位置编码、图结构编码
            learned：文本分类、机器翻译、推荐系统等
        """
        self.embed = 'timeF'  # 嵌入类型 timeF, fixed, learned
        self.freq = 'd'  # 时间频率 [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]


        self.dropout = 0.05 # dropout
        self.num_class = 2  # 分类数


class  ETSformer_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.d_model = 128 # 作为输入嵌入层（DataEmbedding）的输出维度决定Encoder中各层的特征维度 默认为128
        self.n_heads = 8 #  注意力头数，表示多头注意力机制中的头数 需要能够整除d_model，以确保每个头的维度是整数 默认8

        self.d_ff = 256 # 前馈网络维度，表示前馈网络中的隐藏层维度 默认值为256（通常是d_model的2倍）

        self.activation = 'gelu' #  激活函数类型，指定使用的激活函数 默认为gelu
        self.e_layers = 3 # 编码器层数，表示编码器中堆叠的层数 默认为3
        self.top_k = 5 #Top K个显著周期，用于傅里叶变换中选择主要频率成分 默认为5
        #嵌入层参数：嵌入类型 timeF, fixed, learned
        """
            timeF：时间序列预测、时间特征编码
            fixed：NLP中的位置编码、图结构编码
            learned：文本分类、机器翻译、推荐系统等
        """
        self.embed = 'learned'  # 嵌入类型 timeF, fixed, learned 默认：timeF
        self.freq = 'd'  # 时间频率 [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]


        self.dropout = 0.05 # dropout
        self.num_class = 2  # 分类数

class  DLinear_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.moving_avg = 25 # 时间序列分解中移动平均的窗口大小 用于构建移动平均层，提取时间序列的趋势部分


        self.dropout = 0.05 # dropout
        self.num_class = 2  # 分类数

class  TemporalFusionTransformer_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.n_heads = 8 #  注意力头数，表示多头注意力机制中的头数 需要能够整除d_model，以确保每个头的维度是整数 默认8
        self.d_model = 128 # 作为输入嵌入层（DataEmbedding）的输出维度决定Encoder中各层的特征维度 默认为128
        self.static_len = 0 # 静态特征维度
        self.observed_len = int(enc_in/2) # 观测特征维度
        #嵌入层参数：嵌入类型 timeF, fixed, learned
        """
            timeF：时间序列预测、时间特征编码
            fixed：NLP中的位置编码、图结构编码
            learned：文本分类、机器翻译、推荐系统等
        """
        self.embed = 'timeF'  # 嵌入类型 timeF, fixed, learned 默认：timeF
        self.freq = 'd'  # 时间频率 [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]

        self.dropout = 0.05 # dropout
        self.num_class = 2  # 分类数

class  WPMixer_configs:
    def __init__(self,seq_len,enc_in,batch_size=32,device="cpu"):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)
        self.batch_size = batch_size # 批次大小
        self.device = device # 设备名称
        self.patch_len = int(seq_len / 4) # Patch大小,16
        self.use_amp = False # 是否使用混合精度训练

        self.d_model = 128 # 作为输入嵌入层（DataEmbedding）的输出维度决定Encoder中各层的特征维度 默认为128

        self.dropout = 0.05 # dropout
        self.num_class = 2  # 分类数

class  FreTS_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.d_model = 128 # 作为输入嵌入层（DataEmbedding）的输出维度决定Encoder中各层的特征维度 默认为128
        self.channel_independence = '0' # 通道独立性
        self.dropout = 0.05 # dropout
        self.num_class = 2  # 分类数

class  PAttn_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.patch_len = 8 # Patch大小 常见的选择包括 8、12、16、24、32
        self.stride = 2 # 填充长度 通常为 patch_len 的 1/4 到 1/2

        self.d_model = 128 # 作为输入嵌入层（DataEmbedding）的输出维度决定Encoder中各层的特征维度 默认为128
        self.factor = 5 # 注意力机制的缩放因子
        self.n_heads = 8 # 注意力头数，表示多头注意力机制中的头数 需要能够整除d_model，以确保每个头的维度是整数 默认8
        self.d_ff = 256 # 前馈网络维度，表示前馈网络中的隐藏层维度 默认值为256（通常是d_model的2倍）
        self.activation = 'gelu'# 激活函数类型，指定使用的激活函数 默认为gelu
        self.e_layers = 3 # 编码器层数，表示编码器中堆叠的层数 默认为3

        self.dropout = 0.05 # dropout
        self.num_class = 2  # 分类数

class  KANAD_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.d_model = 16 # 作为输入嵌入层（DataEmbedding）的输出维度决定Encoder中各层的特征维度 默认为128
        #self.e_layers = 3 # 编码器层数，表示编码器中堆叠的层数 默认为3

        self.dropout = 0.05 # dropout
        self.num_class = 2  # 分类数

class  Reformer_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.n_heads = 8 #  注意力头数，表示多头注意力机制中的头数 需要能够整除d_model，以确保每个头的维度是整数 默认8
        self.d_model = 128 # 作为输入嵌入层（DataEmbedding）的输出维度决定Encoder中各层的特征维度 默认为128
        self.d_ff = 256 # 前馈网络维度，表示前馈网络中的隐藏层维度 默认值为256（通常是d_model的2倍）
        self.activation = 'gelu'# 激活函数类型，指定使用的激活函数 默认为gelu/relu
        self.e_layers = 3 # 编码器层数，表示编码器中堆叠的层数 默认为3
        #嵌入层参数：嵌入类型 timeF, fixed, learned
        """
            timeF：时间序列预测、时间特征编码
            fixed：NLP中的位置编码、图结构编码
            learned：文本分类、机器翻译、推荐系统等
        """
        self.embed = 'timeF'  # 嵌入类型 timeF, fixed, learned 默认：timeF
        self.freq = 'd'  # 时间频率 [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]

        self.dropout = 0.05 # dropout
        self.num_class = 2  # 分类数

class  Pyraformer_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.n_heads = 32 #  注意力头数，表示多头注意力机制中的头数 需要能够整除d_model，以确保每个头的维度是整数 默认8
        self.d_model = 128 # 作为输入嵌入层（DataEmbedding）的输出维度决定Encoder中各层的特征维度 默认为128
        self.d_ff = 256 # 前馈网络维度，表示前馈网络中的隐藏层维度 默认值为256（通常是d_model的2倍）
        self.activation = 'gelu'# 激活函数类型，指定使用的激活函数 默认为gelu/relu
        self.e_layers = 1 # 编码器层数，表示编码器中堆叠的层数 默认为3
        #嵌入层参数：嵌入类型 timeF, fixed, learned
        """
            timeF：时间序列预测、时间特征编码
            fixed：NLP中的位置编码、图结构编码
            learned：文本分类、机器翻译、推荐系统等
        """
        self.embed = 'timeF'  # 嵌入类型 timeF, fixed, learned 默认：timeF
        self.freq = 'd'  # 时间频率 [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]

        self.dropout = 0.05 # dropout
        self.num_class = 2  # 分类数

class  Crossformer_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.seg_len = 12 # 分段长度
        self.win_size = 4 # 窗口大小

        self.n_heads = 8 #  注意力头数，表示多头注意力机制中的头数 需要能够整除d_model，以确保每个头的维度是整数 默认8
        self.d_model = 128 # 作为输入嵌入层（DataEmbedding）的输出维度决定Encoder中各层的特征维度 默认为128
        self.d_ff = 256 # 前馈网络维度，表示前馈网络中的隐藏层维度 默认值为256（通常是d_model的2倍）

        self.factor = 5 # 注意力机制的缩放因子
        self.e_layers = 1 # 编码器层数，表示编码器中堆叠的层数 默认为3

        self.dropout = 0.05 # dropout
        self.num_class = 2  # 分类数

class  TSMixer_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.d_model = 512 # 表示模型中间层的隐藏维度大小，控制着模型的表达能力和复杂度

        self.e_layers = 5 # 表示堆叠的 ResBlock 残差块数量，决定模型的深度

        self.dropout = 0.05 # dropout
        self.num_class = 2  # 分类数

class  MSGNet_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.d_model = 128 # 表示模型中间层的隐藏维度大小，控制着模型的表达能力和复杂度
        self.d_ff = 256 # 前馈网络维度，表示前馈网络中的隐藏层维度 默认值为256（通常是d_model的2倍）
        self.n_heads = 8 #  注意力头数，表示多头注意力机制中的头数 需要能够整除d_model，以确保每个头的维度是整数 默认8

        self.top_k = 2 # 控制从FFT中提取的主要周期数量，决定多尺度图卷积的层数
        self.c_out = enc_in
        self.conv_channel = 32 # 卷积核数量
        self.skip_channel = 32 # 跳层通道数
        self.gcn_depth = 2 # 图卷积深度
        self.propalpha = 0.3 #传播衰减因子
        self.node_dim = 10 # 节点维度

        self.e_layers = 5 # 多尺度图卷积块：融合图卷积和注意力机制的特征提取模块 默认为3
            # 合理范围：1 - 5层
            # 较浅的网络（1 - 2层）适合简单任务或者数据量较小的情况。
            # 中等深度（3 - 5层）适合更复杂的任务和较大的数据集。
            # 过深可能会导致过拟合，特别是在时间序列分类这种相对任务上。
        #嵌入层参数：嵌入类型 timeF, fixed, learned
        """
            timeF：时间序列预测、时间特征编码
            fixed：NLP中的位置编码、图结构编码
            learned：文本分类、机器翻译、推荐系统等
        """
        self.embed = 'timeF'  # 嵌入类型 timeF, fixed, learned 默认：timeF
        self.freq = 'd'  # 时间频率 [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]


        self.dropout = 0.05 # dropout
        self.num_class = 2  # 分类数


class  TimeFilter_configs:
    def __init__(self,seq_len,enc_in):
        """
        :param seq_len: 时间序列长度
        :param enc_in: 特征数
        """
        self.seq_len = seq_len # 输入序列长度
        self.enc_in = enc_in # 输入特征维度(特征数)

        self.d_model = 128 # 表示模型中间层的隐藏维度大小，控制着模型的表达能力和复杂度
        self.d_ff = 256 # 前馈网络维度，表示前馈网络中的隐藏层维度 默认值为256（通常是d_model的2倍）
        self.patch_len = 8 # 时间序列被切分成的小段（patch）的长度。
        self.pos = True #pos: 是否使用位置编码

        self.n_heads = 8 # 注意力头数，表示多头注意力机制中的头数 需要能够整除d_model，以确保每个头的维度是整数 默认8
        self.e_layers = 3 # 编码器层数，表示编码器中堆叠的层数 默认为3

        self.alpha = 0.1 #控制邻接矩阵稀疏性的阈值比例，在mask_topk函数中用于保留一定比例的最大权重连接。
        self.top_p = 0.5 #在MoE（Mixture of Experts）门控机制中使用的概率累积阈值，用于选择最重要的专家组合。

        self.dropout = 0.05 # dropout
        self.num_class = 2  # 分类数