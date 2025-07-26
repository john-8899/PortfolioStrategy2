# -*- coding: utf-8 -*-
'''
    构建 LSTM 模型
'''
import torch.nn as nn
import torch

'''
    lstm 模型
'''
class LSTM_Model(nn.Module):
    #input_dim 数据特征数据（根据特征数给） hidden_dim隐藏层自定义给，output_dim 输出层
    def __init__(self,input_dim,hidden_dim,num_layers,output_dim):
        super(LSTM_Model,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # lstm层
        self.lstm = nn.LSTM(input_dim,hidden_dim,num_layers,batch_first=True)
        ###Dropout层
        # dropout_rate = 0.1
        # self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers,x.size(0),self.hidden_dim).to(x.device)
        # x 的形状: batch_size,seq_len, input_dim
        out,(hn,cn) = self.lstm(x,(h0.detach(),c0.detach()))
        #out = self.dropout(out)
        out = self.fc(out[:,-1,:])# 只取最后一个时间步的输出
        return out

"""
     双层lstm 模型
"""
class LSTM_Model2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate=0.1):
        super(LSTM_Model2, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 第一层LSTM
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        # 第二层LSTM
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # 第一层LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm1(x, (h0, c0))
        out = self.dropout(out)

        # 第二层LSTM
        h1 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c1 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm2(out, (h1, c1))

        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


'''
    GRU 模型
'''
class GRU_Model(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,output_dim):
        super(GRU_Model,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim,hidden_dim,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        #初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        #向前传播
        out,_ = self.gru(x,h0)
        out = self.fc(out[:,-1,:])
        return out

'''
    BiLSTM
'''
class BiLSTM_Model(nn.Module):
    #input_dim 数据特征数据（根据特征数给） hidden_dim隐藏层自定义给，output_dim 输出层
    def __init__(self,input_dim,hidden_dim,num_layers,output_dim,dropout_rate=0.1):
        super(BiLSTM_Model,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # lstm层
        self.lstm = nn.LSTM(input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,#双向
            batch_first=True)
        ###Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim*2,output_dim)

    def forward(self,x):
        h0 = torch.zeros(2*self.num_layers,x.size(0),self.hidden_dim).to(x.device)
        c0 = torch.zeros(2*self.num_layers,x.size(0),self.hidden_dim).to(x.device)
        # x 的形状: batch_size,seq_len, input_dim
        out,(hn,cn) = self.lstm(x,(h0,c0))
        out = self.dropout(out)
        out = self.fc(out[:,-1,:])# 只取最后一个时间步的输出
        return out


'''
    BiGRU 模型
'''
class BiGRU_Model(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,output_dim,dropout_rate=0.1):
        super(BiGRU_Model,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,#双向
            batch_first=True)
        ###Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim*2,output_dim)

    def forward(self,x):
        #初始化隐藏状态
        h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        #向前传播
        out,_ = self.gru(x,h0)
        out = self.dropout(out)
        out = self.fc(out[:,-1,:])
        return out


"""
    CNN-LSTM
"""

class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, kernel_size,hidden_dim, num_layers,dropout_rate=0.1, output_dim=4):
        super(CNN_LSTM_Model, self).__init__()
        # CNN 层
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),# 池化层
        )

        # LSTM 层参数
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # lstm层
        self.lstm = nn.LSTM(input_size=cnn_out_channels,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # CNN 层
        x = x.permute(0, 2, 1)  # 将输入的形状从 (batch_size, seq_len, input_dim) 转换为 (batch_size, input_dim, seq_len)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # 将输出的形状从 (batch_size, input_dim, seq_len) 转换为 (batch_size, seq_len, input_dim)

        # LSTM 层
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x)

        # Dropout层
        out = self.dropout(out)
        # 全连接层
        out = self.fc(out[:, -1, :])
        return out



"""
    CNN-LSTM-attention
"""

class CNN_LSTM_Attention_Model(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, kernel_size,hidden_dim, num_layers,n_heads,dropout_rate=0.1, output_dim=4):
        super(CNN_LSTM_Attention_Model, self).__init__()
        # CNN 层
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),# 池化层
        )

        # LSTM 层参数
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # lstm层
        self.lstm = nn.LSTM(input_size=cnn_out_channels,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)

        # attention层
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads,batch_first=True)

        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # CNN 层
        x = x.permute(0, 2, 1)  # 将输入的形状从 (batch_size, seq_len, input_dim) 转换为 (batch_size, input_dim, seq_len)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # 将输出的形状从 (batch_size, input_dim, seq_len) 转换为 (batch_size, seq_len, input_dim)
        x = self.dropout(x)
        # LSTM 层
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x)
        # attention层
        out, _ = self.attention(out, out, out)

        # Dropout层
        out = self.dropout(out)
        # 全连接层
        out = self.fc(out[:, -1, :])
        return out