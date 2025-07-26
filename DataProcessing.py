# -*- coding:utf-8 -*-
"""
    数据处理：
        根据原始K线数据，生成技术指标和alpha191数据，以备后续训练使用
"""
from multiprocessing import Pool
from tqdm import tqdm
from technical import Technical
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)#忽略警告
warnings.simplefilter(action='ignore', category=RuntimeWarning)#忽略警告
class CDataProcessing:
    def __init__(self):
        pass

    def GetStockKData(self, code,index=False):
        """
            读取K线数据
        :param code:
        index 是否是指数
        :return:
        """
        path = "D:/DataTransform/tradeData/KlineFQData/%s.csv"%(code)
        if index:
            path = "./DataFiles/%s.csv"%(code)

        datas = pd.read_csv(path)
        featureList = ['nOpen', 'nHigh', 'nLow', 'nClose']
        for feature in featureList:
            datas[feature] = datas[feature]/10000.0
        return datas

    def hand_StockData(self,code,index=False):
        """

        :param code:
        :return:
        """
        print("处理中：%s" % (code))
        kDatas = self.GetStockKData(code, index)
        try:
            # 计算技术指标
            technical = Technical()
            # 计算技术指标
            kDatas = technical.CalIndicator(kDatas)
            # 计算alpha191
            kDatas = technical.CalAlpaha191Factors(kDatas)
            kDatas = kDatas.dropna()
            # 保存数据
            #kDatas.to_csv("D:\DataTransform\indicatorsDatas\KlineFQData\%s.csv"%(code), index=False)
            kDatas.to_csv("./DataFiles/kDatas/%s.csv"%(code),index= False)
            #print("处理完成：%s"%(code))
        except Exception as e:
            print("处理失败：%s"%(code), e)

    def process_data(self):
        """
            处理数据，根据原始K线数据，计算出技术指标和alpha191数据
        :param data:
        :return:
        """
        # 读取股票列表
        CodeList = pd.read_csv("./DataFiles/sh50List.csv",encoding="gbk")

        #多进程处理
        pool = Pool(processes=8)

        pbar = tqdm(total=len(CodeList), desc="处理数据中...")
        def update(*a):
            """
                跟新进度条的回调函数
            :param a:
            :return:
            """
            pbar.update(1)

        for index, row in CodeList.iterrows():
            code = row['nCode']
            pool.apply_async(self.hand_StockData, args=(code,), callback=update)

        pool.close()
        pool.join()
        #处理指数数据
        self.hand_StockData("000016",True)

if __name__ == "__main__":
    #删除DataFiles目录下的所有文件
    import os
    for root, dirs, files in os.walk("./DataFiles/kDatas"):
        for name in files:
            os.remove(os.path.join(root, name))
    #处理数据
    dataProcessing = CDataProcessing()
    dataProcessing.process_data()