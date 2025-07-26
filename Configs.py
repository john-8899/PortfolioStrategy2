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