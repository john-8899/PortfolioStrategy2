"""
    目标：构建一个模型，提前捕捉到未来短期内可能大涨、但不会大跌的股票？
        说得更具体点，我的目标是找出那些在未来5个交易日内：
        涨幅有望超过 5%
        但期间不会回撤超过 3%
        的股票。

        策略建模思路：用机器学习来“择时+选股”
        目标是构建一个二分类模型。我们用历史数据来训练它，使其能判断当前某支股票是否具备这样一种“结构性上涨机会”。

        定义目标标签（Label）
        对于任意一个交易日t，我们设：
        当前收盘价为Pt
        未来5个交易日内的最高价为Pmax = max(Pt+1,.....Pt+5)
        最低价为Pmin = min(Pt+1,...,Pt+5)
        我们的标签定义为：
        Y =
        也就是说：未来5天能涨5%，且没跌超3%，才算正样本。

        以上证50作为案例，后期如果能分出结构性上涨，结合blacklitterman模型使用
"""
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler
from models import *
import pandas as pd
import numpy as np
from EarlyStopping import EarlyStopping
from sklearn.metrics import precision_score,recall_score,f1_score,classification_report,accuracy_score #roc_curve,auc
from collections import Counter
from tqdm import tqdm
from Autoformer import Autoformer
from TimeMixer import TimeMixer
from Nonstationary_Transformer import Nonstationary_Transformer
from Configs import TimeMixer_Configs,TimesNet_configs,Nonstationary_configs,Informer_configs
from TimesNet import CTimesNet
from Informer import Informer
import matplotlib
matplotlib.use('TkAgg')  # 替换当前后端
import matplotlib.pyplot as plt

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 自定义数据集
class StockDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class PortfolioStrategy:
    def __init__(self):
        # 技术指标因子
        self.technicalFactor = ['DIF', 'DEA', 'MACD', 'MA5', 'MA10', 'MA20', 'MA60', 'MA120', 'MA250', 'BOLLUP', 'BOLL',
                                'BOLLDN', 'DEMA5', 'DEMA10', 'DEMA20', 'DEMA60', 'DEMA120', 'DEMA250', 'EMA5', 'EMA10',
                                'EMA20','EMA50', 'EMA200', 'HT_TRENDLINE', 'KAMA10', 'KAMA20', 'KAMA30', 'MAMA', 'FAMA', 'MAVP',
                                'MIDPOINT', 'MIDPRICE', 'SAR', 'T3', 'TRIMA20', 'TRIMA50', 'TRIMA200', 'WMA20', 'WMA50',
                                'WMA200','ADX5', 'ADX14', 'ADX21', 'ADXR', 'APO', 'AROONDOWN', 'AROONUP', 'AROONOSC', 'BOP',
                                'CCI','CMO', 'DX', 'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC',
                                'ROCP','ROCR', 'ROCR100', 'RSI6', 'RSI12', 'RSI24', 'SLOWK', 'SLOWD', 'FASTK', 'FASTD',
                                'fastkRSI', 'fastdRSI','TRIX', 'ULTOSC', 'WILLR', 'AD', 'ADOSC', 'OBV', 'HT_DCPERIOD', 'HT_DCPHASE',
                                'INPHASE','QUADRATURE', 'HT_SINE', 'HT_TRENDMODE', 'ATR', 'NATR', 'TRANGE']
        # alpha191 因子
        self.Alpha191Factors = [f'alpha191_{i + 1}' for i in range(191) if
                                i + 1 not in [30, 75, 122, 149, 164, 173, 181, 182, 190]]

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#这句如果有GPU默认使用GPU0，device = torch.device("cuda:0")# 显式指定第一个GPU
        print(f"Using device: {self.device}")
        if torch.cuda.device_count() > 0:
            print(torch.cuda.device_count(), ":GPUs are available")

    def GetStockKData(self, code):
        path = "./DataFiles/kDatas/%s.csv"%(code)
        datas = pd.read_csv(path)
        return datas

    #训练模式
    def trai_step(self,model,features,labels):
        # 训练模式，dropout层发生作用
        model.train()
        # 梯度清零
        model.optimizer.zero_grad()
        # 正向传播求损失
        predictions = model(features)
        # print(labels.long().squeeze()) 多分类情况下，需要把结果转整型，并展平[32,1]->>[32]
        loss = model.loss_func(predictions,labels.long().squeeze()) #多分类情况下，需要把标签转整型，并展平[32,1]->>[32]
        # 反向传播求梯度
        loss.backward()
        model.optimizer.step()
        return loss.item()#,

    #预测模式
    #@torch.no_grad
    def valid_step(self,model,features,labels):
        #预测模式，dropout层不发生作用
        model.eval()
        predictions = model(features)
        loss = model.loss_func(predictions,labels.long().squeeze())#多分类情况下，需要把标签转整型，并展平[32,1]->>[32]
        return loss.item()#, #metric.item()


    def train_model(self,model,epochs,dl_train,dl_valid,isEarlyStopping=True):
        """
        训练函数
        :param model: 模型
        :param epochs: 训练轮数
        :param dl_train: 训练集数据
        :param dl_valid: 测试集数据
        :param isEarlyStopping: 是否使用早停，True:使用，False:不使用
        :return:
        """
        dfhistory = pd.DataFrame(columns =["epoch","loss","val_loss"])

        # 早停
        early_stop = EarlyStopping(patience=10,verbose=True)

        for epoch in range(1,epochs+1):
            #训练循环----------------------------------------------------------
            loss_sum = 0.0
            metric_sum = 0.0
            step = 1
            for features,labels in dl_train:
                features,labels = features.to(self.device),labels.to(self.device)
                loss = self.trai_step(model,features,labels)
                #打印batch级别日志
                loss_sum += loss
                step+=1

            # #2.验证循环----------------------------------------------------------------------
            val_loss_sum = 0.0
            val_step = 1
            for vfeatures,vlabels in dl_valid:
                vfeatures,vlabels = vfeatures.to(self.device),vlabels.to(self.device)
                val_loss = self.valid_step(model,vfeatures,vlabels)
                val_loss_sum += val_loss
                val_step += 1

            #记录日志
            info = (epoch,loss_sum/step,val_loss_sum/val_step)
            dfhistory.loc[epoch-1] = info
            # 打印epoch级别日志
            print(("EPOCH = %d, loss = %.3f,val_loss = %.3f")%info)
            # 学习率调整
            #scheduler.step(loss_sum/step)

            # 早停机制
            #early_stop(loss_sum/step,model)#训练集losss
            early_stop(val_loss_sum/val_step,model)#验证集losss
            if early_stop.early_stop and isEarlyStopping:
                print("Early stopping")
                model = early_stop.best_model
                break
        return dfhistory

    #预测 在测试集上评估
    def PredictionEvaluation(self, model, testloader):
        """
        评估模型在测试集上的预测效果，返回预测结果和真实标签

        参数:
            model: 待评估的PyTorch模型
            testloader: 测试数据加载器，提供(inputs, labels)数据批次

        返回:
            tuple: (预测结果数组, 真实标签数组)
                   - 预测结果数组: 模型对所有测试样本的预测类别
                   - 真实标签数组: 测试样本对应的真实类别
        """
        precisions_list = []
        labels_list = []

        # 禁用梯度计算以提升评估效率
        with torch.no_grad():
            # 遍历测试集所有批次数据
            for data in testloader:
                inputs, labels = data
                # 将数据转移到指定设备(如GPU)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 模型前向传播获取输出
                outputs = model(inputs)
                # 将概率输出转换为类别预测(取最大概率对应的类别)
                outputs = outputs.argmax(dim=1)

                # 收集当前批次的预测结果和真实标签
                precisions_list.append(outputs.detach().cpu().numpy())
                labels_list.append(labels.squeeze().cpu().numpy())

        # 将所有批次的预测和标签拼接为完整数组
        precisions_list = np.concatenate(precisions_list)
        labels_list = np.concatenate(labels_list)

        return precisions_list, labels_list


    #评估
    def modelEvaluation(self,name,ytest,ypred):
        testAcc = round(accuracy_score(ytest,ypred),3) # 测试集准确率
        fprecision = round(precision_score(ytest,ypred,labels=None,pos_label=1,average='macro',zero_division=1),3) #精确率
        frecall = round(recall_score(ytest,ypred,labels=None,pos_label=1,average='macro',sample_weight=None),3) #召回率
        F1score = round(f1_score(ytest,ypred,labels=None,pos_label=1,average='macro',sample_weight=None),3) #F1
        print("%s,准确率:%0.3f,精确率:%0.3f,召回率;%0.3f,F1core:%0.3f"%(name,testAcc,fprecision,frecall,F1score))
        print("评估报告：\n",classification_report(ytest,ypred,zero_division=1))
        return [testAcc,fprecision,frecall,F1score]

    def plot_metric(self, dfhistory, metric):
        train_metrics = dfhistory[metric]
        #val_metricess = dfhistory["val_" + metric]
        epochs = range(1, len(train_metrics) + 1)
        plt.plot(epochs, train_metrics, "bo--")
        #plt.plot(epochs,val_metricess,"ro--")

        plt.title("Train_" + metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_" + metric, "val_" + metric])
        plt.show()

    def generate_labels(self,code,window=5):
        """
        生成标签列
            参数:
            close_col: 收盘价列名
            window: 未来窗口期(天数)
            up_threshold: 上涨阈值(5%表示为0.05)
            down_threshold: 下跌阈值(-3%表示为-0.03)
        :return:
        """
        datas = pd.read_csv("./DataFiles/kDatas/%s.csv"%(code))
        # 生成标签,标签为t时刻，后5天的最高价相对t涨幅大于5%，未来5天的最低价相对t跌幅<-3%
        #计算未来窗口的期的最高价和最低价
        datas['future_max'] = datas['nHigh'].rolling(window).max().shift(-window)
        datas['future_min'] = datas['nLow'].rolling(window).min().shift(-window)

        #
        datas['max_change'] = (datas['future_max'] - datas['nClose']) / datas['nClose']
        datas['min_change'] = (datas['future_min'] - datas['nClose']) / datas['nClose']

        # 生成标签:标签为1表示涨，0表示跌,涨跌都满足阈值条件则涨
        datas['label'] = np.where((datas['max_change'] >= 0.05)&(datas['min_change']>-0.03),1,0)
        #datas['label'] = np.where((datas['max_change'] >= 0.03)&(datas['min_change']>-0.01),1,0)

        #datas[['nDate', 'nHigh', 'nLow', 'nClose', 'future_max', 'future_min', 'max_change', 'min_change','label']].to_csv("./tmp.csv",index=False)
        # 删除中间计算列
        datas = datas.dropna()
        datas.drop(['future_max', 'future_min', 'max_change', 'min_change'], axis=1, inplace=True)

        # #统计label中每个分类的个数据
        # cdata = Counter(datas['label'])
        # # 分类个数
        # num_calsses = len(cdata)
        # # 总数据量
        # totalCounts = len(datas)
        # cdata = dict(sorted({int(k): v for k, v in cdata.items()}.items()))
        # print("总数据量:",totalCounts,"每个分类的个数统计：",cdata)
        # self.weights = [totalCounts/(num_calsses*v) for k, v in cdata.items()]
        # print("每个分类的分配权重：",self.weights)

        return datas

    def SequenceDivision(self,nLen,featureList,data):
        """
        将时间序列数据划分为训练用的特征序列和对应的标签

        通过滑动窗口的方式，将原始时间序列数据划分为多个连续的子序列作为特征(X)，
        并将每个子序列后一天的数据作为标签(y)。主要用于时间序列预测任务的训练数据准备。

        :param nLen: int, 滑动窗口的长度，即每个特征序列包含的历史数据天数
        :param featureList: list, 需要使用的特征列名列表
        :param data: DataFrame, 包含特征和标签的原始数据集，需包含'label'列
        :return: tuple, (train_x, train_y)
            - train_x: list, 由多个nLen长度的特征序列组成的列表，每个序列是ndarray类型
            - train_y: list, 对应每个特征序列的下一个时间点的标签值
        """
        train_x = []
        train_y = []
        # 使用滑动窗口遍历数据
        for i in range(nLen-1, len(data)):
            # 获取前nLen天的数据(包括当天)
            sequence = data[featureList].iloc[i - nLen+1:i+1].values
            # 获取当天的标签
            label = data['label'].iloc[i]
            train_x.append(sequence)
            train_y.append(label)
        return train_x,train_y

    def makeTrainData(self,nLen,featureList):
        """
            生成训练和测试数据集

            该方法读取股票数据，进行预处理和标准化，并按时间划分训练集和测试集，
            最终生成适用于时间序列模型的输入输出数据。

            参数:
                nLen: int - 序列长度，用于划分时间序列数据

            返回值:
                tuple: 包含四个numpy数组和一个整数
                    - all_train_x: 训练集特征数据
                    - all_train_y: 训练集标签数据
                    - all_test_x: 测试集特征数据
                    - all_test_y: 测试集标签数据
                    - feature_count: 特征数量
        """
        # # 读取上证50成分股列表
        codeList = pd.read_csv("./DataFiles/sh50List.csv",encoding='gbk')

        #初始化存储容器
        all_train_x = []
        all_train_y = []
        all_test_x = []
        all_test_y = []

        ##遍历所有股票代码进行处理
        for index,row in tqdm(codeList.iterrows(),total=len(codeList),desc="数据处理"):
            code = row['nCode']
            #print(f"处理{code}数据")
            # # 生成带标签的数据并处理异常值
            datas = self.generate_labels(code=code,window=5)
            # 添加数据验证，去除无穷大值
            datas = datas.replace([np.inf, -np.inf], 0)

            #根据日期划分训练集和测试集 20240101以前的作为训练集
            trainDatas = datas[datas['nDate']<=20240101].copy()
            # 跳过数据量不足的股票
            if len(trainDatas)<=nLen:
                print(f"{code}训练集，没有数据")
                continue
            testDatas = datas[datas['nDate']>20240101].copy()
            if len(testDatas)<=nLen:
                print(f"{code}测试集，没有数据")
                continue

            #标准化数据
            scaler = MinMaxScaler()
            #标准化训练集
            trainDatas[featureList] = scaler.fit_transform(trainDatas[featureList])
            #标准化测试集
            testDatas[featureList] = scaler.transform(testDatas[featureList])

            # 创建训练集 序列数据
            train_x, train_y = self.SequenceDivision(nLen, featureList, trainDatas)
            # 创建测试集 序列数据
            test_x, test_y = self.SequenceDivision(nLen, featureList, testDatas)

            ## 收集所有股票数据
            all_train_x.extend(train_x)
            all_train_y.extend(train_y)
            all_test_x.extend(test_x)
            all_test_y.extend(test_y)

        # 计算类别权重用于不平衡分类
        #统计label中每个分类的个数据
        cdata = Counter(all_train_y)
        # 分类个数
        num_calsses = len(cdata)
        # 总数据量
        totalCounts = len(all_train_y)
        cdata = dict(sorted({int(k): v for k, v in cdata.items()}.items()))
        print("总数据量:",totalCounts,"每个分类的个数统计：",cdata)
        self.weights = [totalCounts/(num_calsses*v) for k, v in cdata.items()]
        print("每个分类的分配权重：",self.weights)
        return np.array(all_train_x), np.array(all_train_y), np.array(all_test_x), np.array(all_test_y),len(featureList)
    def myMain(self):

        # 合并技术指标因子和Alpha191因子，加上基础行情数据
        featureList = self.technicalFactor  # + self.Alpha191Factors + ['nOpen', 'nHigh', 'nLow', 'nClose', 'iVolume']
        # featureList = ['HT_TRENDMODE', 'DEMA60', 'OBV', 'WILLR', 'TRANGE', 'AROONUP', 'AROONOSC', 'fastdRSI',
        #                'fastkRSI','AROONDOWN', 'ADX5', 'MFI', 'HT_TRENDLINE', 'BOP', 'HT_SINE', 'FASTK', 'DX',
        #                'HT_DCPERIOD','SLOWD','iVolume', 'SLOWK']
        # 生成训练数据(训练集，测试集,特征数)
        nSeqLen = 60
        train_x, train_y,test_x,test_y,featureNum = self.makeTrainData(nLen=nSeqLen, featureList=featureList)
        print("特征数：%d,训练集个数:%d,测试集个数：%d"%(featureNum,len(train_x),len(test_x)))

        # 创建数据集
        train_dataset = StockDataset(train_x, train_y)
        test_dataset = StockDataset(test_x,test_y)

        ##################################处理数据不平衡####################################
        # 获取训练集的类别数量
        class_counts = np.bincount(train_y)
        num_sample = len(train_y)
        # 计算每个类别的权重
        class_weights = 1. / class_counts[train_y]
        weights = torch.from_numpy(class_weights).float()

        #创建 WeightedRandomSampler
        sampler = WeightedRandomSampler(weights, #权重列表
                                        num_samples=num_sample,#样本数量,默认为数据集大小
                                        replacement=True)#允许重复采样
        ##################################处理数据不平衡#######################################

        # 创建数据加载器
        batch_size = 32
        #drop_last = True,如果数据集大小不能被batch_size整除，则丢弃最后一个batch
        #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True,sampler=sampler)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,drop_last=True,sampler=sampler)# sampler时不能用shuffle
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        # 创建模型
        #model = GRU_Model(input_dim=featureNum,hidden_dim=16,num_layers=2,output_dim=2)
        #model = BiGRU_Model(input_dim=featureNum,hidden_dim=64,num_layers=2,output_dim=2)

        #model = Autoformer(input_dim=featureNum,seq_len=nSeqLen,d_model=128,dropout=0.1,moving_avg=5,factor=1.0,output_attention=False,n_heads= 8,d_ff=256,activation='relu',e_layers=3,num_class=2)

        # #TimeMixer模型调用
        # configs = TimeMixer_Configs(seq_len=nSeqLen,enc_in=featureNum)#创建配置对象：TimeMixer模型的参数
        # model = TimeMixer(configs)

        # TimesNet模型调用
        # configs = TimesNet_configs(seq_len=nSeqLen,enc_in=featureNum)
        # model = CTimesNet(configs)

        # # Nonstationary_Transformer模型调用 #序列长度，输入特征数
        # configs = Nonstationary_configs(seq_len=nSeqLen,enc_in=featureNum)
        # model = Nonstationary_Transformer(configs)

        # Informer模型调用
        configs = Informer_configs(seq_len=nSeqLen,enc_in=featureNum)
        model = Informer(configs)

        model = model.to(self.device)
        #设置优化器
        #weight_decay 正则化参数
        #model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        model.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=2.3208e-5)

        # 设置损失函数
        # 对数据不均衡 重新设置分类权重(self.weights 在：makeTrainData 函数计算)
        # weights = torch.tensor(self.weights, dtype=torch.float32)
        # model.loss_func = nn.CrossEntropyLoss(weight=weights)
        model.loss_func = nn.CrossEntropyLoss().to(self.device)

        # 设置训练轮数
        epochs = 1200
        # 训练模型:
        dfhistory = self.train_model(model,epochs,train_loader,test_loader,isEarlyStopping=True)

        # 训练集上评估模型
        xpred,xtest = self.PredictionEvaluation(model,train_loader)
        self.modelEvaluation("训练集上评估模型",xtest,xpred)

        # 测试集上评估模型
        ypred,ytest = self.PredictionEvaluation(model,test_loader)
        self.modelEvaluation("测试集上评估模型",ytest,ypred)
        # 绘制训练曲线
        self.plot_metric(dfhistory, "loss")

        #保存模型到SavedModel目录下(保存的是模型的参数)
        torch.save(model, "SaveModels/GRU_model.pth")

if __name__ == '__main__':
    ps = PortfolioStrategy()
    ps.myMain()