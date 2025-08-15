# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFilter(nn.Module):
    """
    Fourier Filter: to time-variant and time-invariant term
    """

    def __init__(self, mask_spectrum):
        super(FourierFilter, self).__init__()
        self.mask_spectrum = mask_spectrum

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        mask = torch.ones_like(xf)
        mask[:, self.mask_spectrum, :] = 0
        x_var = torch.fft.irfft(xf * mask, dim=1)
        x_inv = x - x_var

        return x_var, x_inv


class MLP(nn.Module):
    '''
        Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''

    def __init__(self,
                 f_in,
                 f_out,
                 hidden_dim=128,
                 hidden_layers=2,
                 dropout=0.05,
                 activation='tanh'):
        super(MLP, self).__init__()
        self.f_in:int = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim),
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers - 2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]

        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y


class KPLayer(nn.Module):
    """
    A demonstration of finding one step transition of linear system by DMD iteratively
    """

    def __init__(self):
        super(KPLayer, self).__init__()

        self.K = None  # B E E

    def one_step_forward(self, z, return_rec=False, return_K=False):
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution  # B E E
        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_pred = torch.bmm(z[:, -1:], self.K)
        if return_rec:
            z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
            return z_rec, z_pred

        return z_pred

    def forward(self, z, pred_len=1):
        assert pred_len >= 1, 'prediction length should not be less than 1'
        z_rec, z_pred = self.one_step_forward(z, return_rec=True)
        z_preds = [z_pred]
        for i in range(1, pred_len):
            z_pred = torch.bmm(z_pred, self.K)
            z_preds.append(z_pred)
        z_preds = torch.cat(z_preds, dim=1)
        return z_rec, z_preds


class KPLayerApprox(nn.Module):
    """
    Find koopman transition of linear system by DMD with multistep K approximation
    """

    def __init__(self):
        super(KPLayerApprox, self).__init__()

        self.K = None  # B E E
        self.K_step = None  # B E E

    def forward(self, z, pred_len=1):
        # z:       B L E, koopman invariance space representation
        # z_rec:   B L E, reconstructed representation
        # z_pred:  B S E, forecasting representation
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution  # B E E

        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)  # B L E

        if pred_len <= input_len:
            self.K_step = torch.linalg.matrix_power(self.K, pred_len)
            if torch.isnan(self.K_step).any():
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
            z_pred = torch.bmm(z[:, -pred_len:, :], self.K_step)
        else:
            self.K_step = torch.linalg.matrix_power(self.K, input_len)
            if torch.isnan(self.K_step).any():
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
            temp_z_pred, all_pred = z, []
            for _ in range(math.ceil(pred_len / input_len)):
                temp_z_pred = torch.bmm(temp_z_pred, self.K_step)
                all_pred.append(temp_z_pred)
            z_pred = torch.cat(all_pred, dim=1)[:, :pred_len, :]

        return z_rec, z_pred


class TimeVarKP(nn.Module):
    """
    Koopman Predictor with DMD (analysitical solution of Koopman operator)
    Utilize local variations within individual sliding window to predict the future of time-variant term
    """

    def __init__(self,
                 enc_in=8,
                 input_len=96,
                 pred_len=96,
                 seg_len=24,
                 dynamic_dim=128,
                 encoder=None,
                 decoder=None,
                 multistep=False,
                 ):
        super(TimeVarKP, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        self.dynamic_dim = dynamic_dim
        self.multistep = multistep
        self.encoder, self.decoder = encoder, decoder
        self.freq = math.ceil(self.input_len / self.seg_len)  # segment number of input
        self.step = math.ceil(self.pred_len / self.seg_len)  # segment number of output
        self.padding_len = self.seg_len * self.freq - self.input_len
        # Approximate mulitstep K by KPLayerApprox when pred_len is large
        self.dynamics = KPLayerApprox() if self.multistep else KPLayer()

    def forward(self, x):
        # x: B L C
        B, L, C = x.shape

        res = torch.cat((x[:, L - self.padding_len:, :], x), dim=1)

        res = res.chunk(self.freq, dim=1)  # F x B P C, P means seg_len
        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)  # B F PC

        res = self.encoder(res)  # B F H
        x_rec, x_pred = self.dynamics(res, self.step)  # B F H, B S H

        x_rec = self.decoder(x_rec)  # B F PC
        x_rec = x_rec.reshape(B, self.freq, self.seg_len, self.enc_in)
        x_rec = x_rec.reshape(B, -1, self.enc_in)[:, :self.input_len, :]  # B L C

        x_pred = self.decoder(x_pred)  # B S PC
        x_pred = x_pred.reshape(B, self.step, self.seg_len, self.enc_in)
        x_pred = x_pred.reshape(B, -1, self.enc_in)[:, :self.pred_len, :]  # B S C

        return x_rec, x_pred


class TimeInvKP(nn.Module):
    """
    Koopman Predictor with learnable Koopman operator
    Utilize lookback and forecast window snapshots to predict the future of time-invariant term
    """

    def __init__(self,
                 input_len=96,
                 pred_len=96,
                 dynamic_dim=128,
                 encoder=None,
                 decoder=None):
        super(TimeInvKP, self).__init__()
        self.dynamic_dim = dynamic_dim
        self.input_len = input_len
        self.pred_len = pred_len
        self.encoder = encoder
        self.decoder = decoder

        K_init = torch.randn(self.dynamic_dim, self.dynamic_dim)
        U, _, V = torch.svd(K_init)  # stable initialization
        self.K = nn.Linear(self.dynamic_dim, self.dynamic_dim, bias=False)
        self.K.weight.data = torch.mm(U, V.t())

    def forward(self, x):
        # x: B L C
        res = x.transpose(1, 2)  # B C L
        res = self.encoder(res)  # B C H
        res = self.K(res)  # B C H
        res = self.decoder(res)  # B C S
        res = res.transpose(1, 2)  # B S C

        return res


class Koopa(nn.Module):
    """Koopa模型：专门用于时间序列分类任务的模型
    Args:
        configs: 模型配置参数，包含seq_len, num_class等
    """

    #def __init__(self, configs, dynamic_dim=128, hidden_dim=64, hidden_layers=2, num_blocks=3, multistep=False):
    def __init__(self, configs):
        """
        mask_spectrum: list, shared frequency spectrums 共享频率谱
        seg_len: int, segment length of time series 时间序列的段长度
        dynamic_dim: int, latent dimension of koopman embedding 嵌入的潜在维度
        hidden_dim: int, hidden dimension of en/decoder  编码器/解码器的隐藏维度
        hidden_layers: int, number of hidden layers of en/decoder 编码器/解码器的隐藏层数量
        num_blocks: int, number of Koopa blocks Koopa块的数量
        multistep: bool, whether to use approximation for multistep K  否使用多步K的近似
        alpha: float, spectrum filter ratio 光谱滤波比
        """
        super(Koopa, self).__init__()
        self.configs = configs
        self.seq_len:int = configs.seq_len
        self.num_class:int = configs.num_class
        self.enc_in:int = configs.enc_in

        # self.seg_len = 24  # 固定segment长度用于分类任务
        # self.num_blocks = num_blocks
        # self.dynamic_dim = dynamic_dim
        # self.hidden_dim = hidden_dim
        # self.hidden_layers = hidden_layers
        # self.multistep = multistep
        # self.alpha = 0.2

        self.seg_len = 24  # 固定segment长度用于分类任务 默认：24
        self.num_blocks:int = configs.num_blocks
        self.dynamic_dim:int = configs.dynamic_dim
        self.hidden_dim:int = configs.hidden_dim
        self.hidden_layers:int = configs.hidden_layers
        self.multistep: bool = configs.multistep
        self.alpha: float = configs.alpha

        # 为分类任务简化mask_spectrum计算
        self.mask_spectrum = torch.arange(0, int(self.seq_len / 2 * self.alpha))

        self.disentanglement = FourierFilter(self.mask_spectrum)

        # shared encoder/decoder to make koopman embedding consistent
        self.time_inv_encoder = MLP(f_in=self.seq_len, f_out=self.dynamic_dim, activation='relu',
                                    hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_inv_decoder = MLP(f_in=self.dynamic_dim, f_out=self.seq_len, activation='relu',
                                    hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_inv_kps = nn.ModuleList([
            TimeInvKP(input_len=self.seq_len,
                      pred_len=self.seq_len,
                      dynamic_dim=self.dynamic_dim,
                      encoder=self.time_inv_encoder,
                      decoder=self.time_inv_decoder)
            for _ in range(self.num_blocks)])

        # shared encoder/decoder to make koopman embedding consistent
        self.time_var_encoder = MLP(f_in=self.seg_len * self.enc_in, f_out=self.dynamic_dim, activation='tanh',
                                    hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_var_decoder = MLP(f_in=self.dynamic_dim, f_out=self.seg_len * self.enc_in, activation='tanh',
                                    hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_var_kps = nn.ModuleList([
            TimeVarKP(enc_in=configs.enc_in,
                      input_len=self.seq_len,
                      pred_len=self.seq_len,
                      seg_len=self.seg_len,
                      dynamic_dim=self.dynamic_dim,
                      encoder=self.time_var_encoder,
                      decoder=self.time_var_decoder,
                      multistep=self.multistep)
            for _ in range(self.num_blocks)])

        # 分类任务专用投影层
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        #计算时间特维度
        self.feature_dim = self.seq_len * self.enc_in * self.num_blocks * 2 # 乘以2因为有time_inv和time_var两个分支
        self.projection = nn.Linear(self.feature_dim,configs.num_class)

    def forward(self, x_enc, x_mark_enc=None):
        """前向传播函数：处理输入序列并输出分类结果

        Args:
            x_enc: 输入序列数据，形状为 [B, T, C]
            x_mark_enc: 时间标记数据，用于掩码填充部分，形状为 [B, T]

        Returns:
            output: 分类结果，形状为 [B, num_class]
        """
        # Koopman特征提取
        features = []
        residual = x_enc
        for i in range(self.num_blocks):
            time_var_input, time_inv_input = self.disentanglement(residual)
            time_inv_output = self.time_inv_kps[i](time_inv_input)
            time_var_backcast, time_var_output = self.time_var_kps[i](time_var_input)
            residual = residual - time_var_backcast

            # 将time_inv和time_var的输出展平并连接作为特征
            time_inv_flat = time_inv_output.reshape(time_inv_output.shape[0], -1)
            time_var_flat = time_var_output.reshape(time_var_output.shape[0], -1)
            features.append(time_inv_flat)
            features.append(time_var_flat)

        # 连接所有块的特征
        features = torch.cat(features, dim=1)  # [B, seq*e_inc * num_blocks * 2]

        # 分类头处理
        output = self.act(features)
        output = self.dropout(output)
        output = self.projection(output)  # [B, num_class]
        return output