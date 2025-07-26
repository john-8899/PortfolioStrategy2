# -*- coding: utf-8 -*-
__author__ = 'tom'
'''
    技术指标计算模块
'''
import talib as tab
from AlphaFactor.Alpha191 import *

class Technical:
    def __init__(self):
        featureList = ['DIF', 'DEA', 'MACD', 'MA5', 'MA10', 'MA20', 'MA60', 'MA120', 'MA250', 'BOLLUP', 'BOLL', 'BOLLDN',
                    'DEMA5', 'DEMA10', 'DEMA20', 'DEMA60', 'DEMA120', 'DEMA250', 'EMA5', 'EMA10', 'EMA20', 'EMA50',
                    'EMA200', 'HT_TRENDLINE', 'KAMA10', 'KAMA20', 'KAMA30', 'MAMA', 'FAMA', 'MAVP', 'MIDPOINT',
                    'MIDPRICE', 'SAR', 'T3', 'TRIMA20', 'TRIMA50', 'TRIMA200', 'WMA20', 'WMA50', 'WMA200', 'ADX5',
                    'ADX14', 'ADX21', 'ADXR', 'APO', 'AROONDOWN', 'AROONUP', 'AROONOSC', 'BOP', 'CCI', 'CMO', 'DX',
                    'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100',
                    'RSI6', 'RSI12', 'RSI24', 'SLOWK', 'SLOWD', 'FASTK', 'FASTD', 'fastkRSI', 'fastdRSI', 'TRIX',
                    'ULTOSC', 'WILLR', 'AD', 'ADOSC', 'OBV', 'HT_DCPERIOD', 'HT_DCPHASE', 'INPHASE', 'QUADRATURE',
                    'HT_SINE', 'HT_TRENDMODE', 'ATR', 'NATR', 'TRANGE']

        #print(f'特征数：{len(featureList)}')
    def CalIndicator(self,kDatas):
        """
            技术指标因子
        :param kDatas:
        :return:
        """
        # 计算涨跌幅度
        kDatas["return"] = kDatas["nClose"].pct_change()
        #################################################### 均线指标 #######################################################
        #简单均线
        kDatas['MA5'] = tab.MA(kDatas['nClose'].values,timeperiod=5)
        kDatas['MA10'] = tab.MA(kDatas['nClose'].values,timeperiod=10)
        kDatas['MA20'] = tab.MA(kDatas['nClose'].values,timeperiod=20)
        kDatas['MA60'] = tab.MA(kDatas['nClose'].values,timeperiod=60)
        kDatas['MA120'] = tab.MA(kDatas['nClose'].values,timeperiod=120)
        kDatas['MA250'] = tab.MA(kDatas['nClose'].values,timeperiod=250)

        # BOLL 布林 参数20日
        kDatas['BOLLUP'], kDatas["BOLL"], kDatas['BOLLDN'] = tab.BBANDS(kDatas["nClose"].values, timeperiod=20,nbdevup=2.0, nbdevdn=2.0)

        #DEMA 双移动均线
        kDatas["DEMA5"] = tab.DEMA(kDatas['nClose'].values, timeperiod=5)
        kDatas["DEMA10"] = tab.DEMA(kDatas['nClose'].values, timeperiod=10)
        kDatas["DEMA20"] = tab.DEMA(kDatas['nClose'].values, timeperiod=20)
        kDatas["DEMA60"] = tab.DEMA(kDatas['nClose'].values, timeperiod=60)
        kDatas["DEMA120"] = tab.DEMA(kDatas['nClose'].values, timeperiod=120)
        kDatas["DEMA250"] = tab.DEMA(kDatas['nClose'].values, timeperiod=250)

        #EMA  指数移动平均
        kDatas["EMA5"] = tab.EMA(kDatas['nClose'].values, timeperiod=5)
        kDatas["EMA10"] = tab.EMA(kDatas['nClose'].values, timeperiod=10)
        kDatas["EMA20"] = tab.EMA(kDatas['nClose'].values, timeperiod=20)
        kDatas["EMA50"] = tab.EMA(kDatas['nClose'].values, timeperiod=50)
        kDatas["EMA200"] = tab.EMA(kDatas['nClose'].values, timeperiod=200)

        #HT_TRENDLINE希尔伯特瞬时变换
        kDatas["HT_TRENDLINE"] = tab.HT_TRENDLINE(kDatas['nClose'].values)

        #KAMA KAMA考夫曼的自适应移动平均线
        kDatas["KAMA10"] = tab.KAMA(kDatas['nClose'].values, timeperiod=10)
        kDatas["KAMA20"] = tab.KAMA(kDatas['nClose'].values, timeperiod=20)
        kDatas["KAMA30"] = tab.KAMA(kDatas['nClose'].values, timeperiod=30)

        #MAMA自适应移动平均线
        kDatas["MAMA"], kDatas["FAMA"] = tab.MAMA(kDatas['nClose'].values)

        #MAVP 多参数移动平均线
        kDatas["MAVP"] = tab.MAVP(kDatas['nClose'].values, kDatas['nClose'].values, minperiod=2, maxperiod=30, matype=0)

        #MIDPOINT - 中期点
        kDatas["MIDPOINT"] = tab.MIDPOINT(kDatas['nClose'].values, timeperiod=14)
        #MIDPRICE - 中期价
        kDatas["MIDPRICE"] = tab.MIDPRICE(kDatas["nHigh"].values, kDatas["nLow"].values, timeperiod=14)

        ##SAR
        kDatas["SAR"] = tab.SAR(kDatas["nHigh"].values, kDatas["nLow"].values, acceleration=0.02, maximum=0.2)

        #T3三重指数移动平均线
        kDatas["T3"] = tab.T3(kDatas['nClose'].values, timeperiod=5, vfactor=0)

        #TRIMA - 三角移动平均线
        kDatas["TRIMA20"] = tab.TRIMA(kDatas['nClose'].values, timeperiod=20)
        kDatas["TRIMA50"] = tab.TRIMA(kDatas['nClose'].values, timeperiod=50)
        kDatas["TRIMA200"] = tab.TRIMA(kDatas['nClose'].values, timeperiod=200)

        #WMA - 加权移动平均线
        kDatas["WMA20"] = tab.WMA(kDatas['nClose'].values, timeperiod=20)
        kDatas["WMA50"] = tab.WMA(kDatas['nClose'].values, timeperiod=50)
        kDatas["WMA200"] = tab.WMA(kDatas['nClose'].values, timeperiod=200)

        ############################### 动量指标 ##################################################
        # ADX ADX平均趋向指数
        kDatas["ADX5"] = tab.ADX(kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values, timeperiod=5)
        kDatas["ADX14"] = tab.ADX(kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values, timeperiod=14)
        kDatas["ADX21"] = tab.ADX(kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values, timeperiod=21)

        #ADXR平均趋向指数的趋向指数
        kDatas["ADXR"] = tab.ADXR(kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values, timeperiod=14)

        #APO - 绝对价格振荡器
        kDatas["APO"] = tab.APO(kDatas['nClose'].values, fastperiod=12, slowperiod=26, matype=0)

        #AROON阿隆指标
        kDatas["AROONDOWN"], kDatas["AROONUP"] = tab.AROON(kDatas["nHigh"].values, kDatas["nLow"].values, timeperiod=14)

        #AROONOSC阿隆振荡
        kDatas["AROONOSC"] = tab.AROONOSC(kDatas["nHigh"].values, kDatas["nLow"].values, timeperiod=14)

        #BOP 均势指标
        kDatas["BOP"] = tab.BOP(kDatas["nOpen"].values, kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values)

        #CCI顺势指标
        kDatas["CCI"] = tab.CCI(kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values, timeperiod=14)

        #CMO CMO钱德动量摆动指标
        kDatas["CMO"] = tab.CMO(kDatas['nClose'].values, timeperiod=14)

        # DX DX动向指标或趋向指标
        kDatas["DX"] = tab.DX(kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values, timeperiod=14)

        #MFI资金流量指标
        kDatas["MFI"] = tab.MFI(kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values, kDatas["iVolume"].astype(float).values, timeperiod=14)

        #MINUS_DI下升动向值(与DX相似)
        kDatas["MINUS_DI"] = tab.MINUS_DI(kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values, timeperiod=14)

        #MINUS_DM上升动向值（与DX相似）
        kDatas["MINUS_DM"] = tab.MINUS_DM(kDatas["nHigh"].values, kDatas["nLow"].values, timeperiod=14)

        #MOM上升动向值
        kDatas["MOM"] = tab.MOM(kDatas['nClose'].values, timeperiod=12)

        # PLUS_DI - Plus方向指示器
        kDatas["PLUS_DI"] = tab.PLUS_DI(kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values, timeperiod=14)

        #PLUS_DM - Plus定向运动
        kDatas["PLUS_DM"] = tab.PLUS_DM(kDatas["nHigh"].values, kDatas["nLow"].values, timeperiod=14)

        #PPO PPO价格震荡百分比指数
        kDatas["PPO"] = tab.PPO(kDatas['nClose'].values, fastperiod=12, slowperiod=26, matype=0)

        # ROC ROC变动率指标
        kDatas["ROC"] = tab.ROC(kDatas['nClose'].values, timeperiod=12)

        #ROCP - 变化率百分比
        kDatas["ROCP"] = tab.ROCP(kDatas['nClose'].values, timeperiod=12)

        #ROCR - 变化率比率
        kDatas["ROCR"] = tab.ROCR(kDatas['nClose'].values, timeperiod=12)

        #ROCR100 - 变化率100比例
        kDatas["ROCR100"] = tab.ROCR100(kDatas['nClose'].values, timeperiod=12)

        # RSI 参数 6日
        kDatas["RSI6"] = tab.RSI(kDatas['nClose'].values, timeperiod=6)
        kDatas["RSI12"] = tab.RSI(kDatas['nClose'].values, timeperiod=12)
        kDatas["RSI24"] = tab.RSI(kDatas['nClose'].values, timeperiod=24)

        #SLOWK -STOCH - Stochastic KDJ指标中的KD指标
        kDatas["SLOWK"], kDatas["SLOWD"] = tab.STOCH(kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

        #STOCHF - 随机快速
        kDatas['FASTK'],  kDatas['FASTD'] =tab.STOCHF(kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values, fastk_period=5, fastd_period=3, fastd_matype=0)

        #STOCHRSI - STOCHRSI - 随机相对强弱指数
        kDatas['fastkRSI'],kDatas['fastdRSI'] = tab.STOCHRSI(kDatas['nClose'].values, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)

        ###TRIX 三次平滑EMA的1天变化率（与ROC相似）
        kDatas["TRIX"] = tab.TRIX(kDatas['nClose'].values, timeperiod=12)

        #ULTOSC终极波动指标 一种多方位功能的指标，除了趋势确认及超买超卖方面的作用之外，它的“突破”讯号不仅可以提供最适当的交易时机之外，更可以进一步加强指标的可靠度。
        kDatas["ULTOSC"] = tab.ULTOSC(kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        #WILLR Williams %R指标
        kDatas["WILLR"] = tab.WILLR(kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values, timeperiod=14)


        ##############################################交易量指标############################################
        #  AD - 累积分布线
        kDatas["AD"] = tab.AD(kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values, kDatas["iVolume"].astype(float).values)

        #  ADOSC - 累积分布线 oscillator 将资金流动情况与价格行为相对比，检测市场中资金流入和流出的情况
        kDatas["ADOSC"] = tab.ADOSC(kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values, kDatas["iVolume"].astype(float).values, fastperiod=3, slowperiod=10)

        #  OBV - 累积交易量指标
        kDatas["OBV"] = tab.OBV(kDatas["nClose"].values, kDatas["iVolume"].astype(float).values)


        ################################################ 周期指标 ##########################################################
        #HT_DCPERIOD希尔伯特变换-主导周期
        kDatas["HT_DCPERIOD"] = tab.HT_DCPERIOD(kDatas["nClose"].values)

        #HT_DCPHASE希尔伯特变换-主导循环阶段
        kDatas["HT_DCPHASE"] = tab.HT_DCPHASE(kDatas["nClose"].values)

        #HT_ PHASOR希尔伯特变换-希尔伯特变换相量分量
        kDatas["INPHASE"], kDatas["QUADRATURE"] = tab.HT_PHASOR(kDatas["nClose"].values)

        #HT_ SINE希尔伯特变换-正弦波
        kDatas["HT_SINE"], kDatas["HT_TRENDLINE"] = tab.HT_SINE(kDatas["nClose"].values)

        #HT_ TRENDMODE希尔伯特变换-趋势与周期模式
        kDatas["HT_TRENDMODE"] = tab.HT_TRENDMODE(kDatas["nClose"].values)

        ############################################## 波动率指标 ########################################################
        #ATR真实波动幅度均值
        kDatas["ATR"] = tab.ATR(kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values, timeperiod=14)

        #NATR归一化波动幅度均值
        kDatas["NATR"] = tab.NATR(kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values, timeperiod=14)

        #TRANGE真正的范围
        kDatas["TRANGE"] = tab.TRANGE(kDatas["nHigh"].values, kDatas["nLow"].values, kDatas["nClose"].values)

        #去掉Nan值
        # kDatas = kDatas.dropna()

        return kDatas

    def CalAlpaha191Factors(self, kData):
        """
            alpha191 因子计算
        :param kData:
        :return:
        """
        kDatas = kData.copy()
        #重置索引
        kDatas.reset_index(inplace=True,drop=True)
        close_df = kDatas['nClose'].copy()
        open_df = kDatas['nOpen'].copy()
        high_df = kDatas['nHigh'].copy()
        low_df = kDatas['nLow'].copy()
        volume_df = kDatas['iVolume'].copy().astype(np.float64)
        amount_df = kDatas['iTurover'].copy()
        vwap_df = vwap(close_df, volume_df)

        kDatas["alpha191_1"] = alpha191_1(volume_df, open_df, close_df, n=6)
        kDatas["alpha191_2"] = alpha191_2(high_df,low_df, close_df)
        kDatas["alpha191_3"] = alpha191_3(close_df,high_df,low_df,n=6)
        kDatas["alpha191_4"] = alpha191_4(volume_df, close_df, n=8,m=20)
        kDatas["alpha191_5"] = alpha191_5(volume_df, high_df,n=5,m=3)
        kDatas["alpha191_6"] = alpha191_6(open_df,high_df,n=4)
        kDatas["alpha191_7"] = alpha191_7(volume_df, close_df, vwap_df, n=3)
        kDatas["alpha191_8"] = alpha191_8(high_df,low_df,vwap_df,n=4)
        kDatas["alpha191_9"] = alpha191_9(high_df,low_df,volume_df)
        kDatas["alpha191_10"] = alpha191_10(close_df,ret_df=None)

        kDatas["alpha191_11"] = alpha191_11(close_df,high_df,low_df,volume_df,n=6)
        kDatas["alpha191_12"] = alpha191_12(open_df,close_df,vwap_df)
        kDatas["alpha191_13"] = alpha191_13(high_df,low_df,vwap_df)
        kDatas["alpha191_14"] = alpha191_14(close_df,n=5)
        kDatas["alpha191_15"] = alpha191_15(open_df,close_df)
        kDatas["alpha191_16"] = alpha191_16(volume_df, vwap_df, n=5)
        kDatas["alpha191_17"] = alpha191_17(vwap_df, close_df, n=5)
        kDatas["alpha191_18"] = alpha191_18(close_df, n=5)
        kDatas["alpha191_19"] = alpha191_19(close_df, n=5)
        kDatas["alpha191_20"] = alpha191_20(close_df, n=6)
        kDatas["alpha191_21"] = alpha191_21(close_df, n=5)
        kDatas["alpha191_22"] = alpha191_22(close_df, n=6)
        kDatas["alpha191_23"] = alpha191_23(close_df, n=20)
        kDatas["alpha191_24"] = alpha191_24(close_df, n=5)
        kDatas["alpha191_25"] = alpha191_25(close_df,ret_df=None,volume_df=volume_df, n=250)
        kDatas["alpha191_26"] = alpha191_26(close_df,vwap_df ,n=230)
        kDatas["alpha191_27"] = alpha191_27(close_df, n=12)
        kDatas["alpha191_28"] = alpha191_28(close_df,high_df, low_df, n=9)
        kDatas["alpha191_29"] = alpha191_29(close_df,volume_df ,n=6)
        #####kDatas["alpha191_30"] = alpha191_30(close_df, n=5)
        kDatas["alpha191_31"] = alpha191_31(close_df, n=12)
        kDatas["alpha191_32"] = alpha191_32(high_df,volume_df, n=3)
        kDatas["alpha191_33"] = alpha191_33(close_df,low_df,volume_df, ret_df=None,n=5,m=240)
        kDatas["alpha191_34"] = alpha191_34(close_df, n=12)
        kDatas["alpha191_35"] = alpha191_35(open_df,volume_df, n=15,m=17)
        kDatas["alpha191_36"] = alpha191_36(volume_df,vwap_df, n=6)
        kDatas["alpha191_37"] = alpha191_37(open_df, ret_df=None, n=5,m=10)
        kDatas["alpha191_38"] = alpha191_38(high_df, n=20)
        kDatas["alpha191_39"] = alpha191_39(close_df,open_df,volume_df, vwap_df, n=8,m=14,l=12)
        kDatas["alpha191_40"] = alpha191_40(close_df, volume_df,n=26)
        kDatas["alpha191_41"] = alpha191_41(vwap_df,n=5)
        kDatas["alpha191_42"] = alpha191_42(high_df,volume_df, n=10)
        kDatas["alpha191_43"] = alpha191_43(close_df,volume_df,n=6)
        kDatas["alpha191_44"] = alpha191_44(low_df,volume_df,vwap_df, n=10,m=7)
        kDatas["alpha191_45"] = alpha191_45(close_df,open_df,volume_df,vwap_df,n=15)
        kDatas["alpha191_46"] = alpha191_46(close_df,volume_df,n=3)
        kDatas["alpha191_47"] = alpha191_47(close_df,high_df,low_df,n=6,m=9)
        kDatas["alpha191_48"] = alpha191_48(close_df,volume_df,n=5,m=20)
        kDatas["alpha191_49"] = alpha191_49(high_df,low_df,n=12)
        kDatas = kDatas.copy() # 防止数据碎片化
        kDatas["alpha191_50"] = alpha191_50(high_df,low_df,n=12)
        kDatas["alpha191_51"] = alpha191_51(high_df,low_df,n=12)
        kDatas["alpha191_52"] = alpha191_52(close_df,high_df,low_df,n=26)
        kDatas["alpha191_53"] = alpha191_53(close_df,change_flag=False,n=12)
        kDatas["alpha191_54"] = alpha191_54(close_df,open_df,n=10)
        kDatas["alpha191_55"] = alpha191_55(close_df,open_df,high_df,low_df,n=20)
        kDatas["alpha191_56"] = alpha191_56(open_df,high_df,low_df,volume_df,n1=12,n2=19,n3=40,n4=13)
        kDatas["alpha191_57"] = alpha191_57(close_df,high_df,low_df,n=9)
        kDatas["alpha191_58"] = alpha191_58(close_df,change_flag=False,n=20)
        kDatas["alpha191_59"] = alpha191_59(close_df,high_df,low_df,n=20)
        kDatas["alpha191_60"] = alpha191_60(close_df,high_df,low_df,volume_df,n=20)
        kDatas["alpha191_61"] = alpha191_61(vwap_df,low_df,volume_df,n1=80,n2=8,n3=12,n4=17)
        kDatas["alpha191_62"] = alpha191_62(high_df,volume_df,n=5)
        kDatas["alpha191_63"] = alpha191_63(close_df,n=6)
        kDatas["alpha191_64"] = alpha191_64(close_df,volume_df,n1=60,n2=4,n3=13,n4=14)
        kDatas["alpha191_65"] = alpha191_65(close_df,n=6)
        kDatas["alpha191_66"] = alpha191_66(close_df,n=6)
        kDatas["alpha191_67"] = alpha191_67(close_df,n=24)
        kDatas["alpha191_68"] = alpha191_68(high_df,low_df,volume_df,n=15)
        kDatas["alpha191_69"] = alpha191_69(close_df,open_df,n=20)
        kDatas["alpha191_70"] = alpha191_70(amount_df,n=6)
        kDatas["alpha191_71"] = alpha191_71(close_df,n=24)
        kDatas["alpha191_72"] = alpha191_72(close_df,high_df,low_df,n=15)
        kDatas["alpha191_73"] = alpha191_73(close_df,volume_df,vwap_df,n1=10,n2=16,n3=4,n4=5,n5=3)
        kDatas["alpha191_74"] = alpha191_74(low_df,volume_df,vwap_df,n1=20,n2=40,n3=7,n4=6)
        ###kDatas["alpha191_75"] = alpha191_75(close_df,open_df,benchmark_close_df,benchmark_open_df,n=50)
        kDatas["alpha191_76"] = alpha191_76(close_df,volume_df,n1=20,n2=20)
        kDatas["alpha191_77"] = alpha191_77(high_df,low_df,volume_df,vwap_df,n1=20,n2=3,n3=6)
        kDatas["alpha191_78"] = alpha191_78(close_df,high_df,low_df,n=12)
        kDatas["alpha191_79"] = alpha191_79(close_df,n=12)
        kDatas["alpha191_80"] = alpha191_80(volume_df,n=5)
        kDatas["alpha191_81"] = alpha191_81(volume_df,n=21)
        kDatas["alpha191_82"] = alpha191_82(close_df,high_df,low_df,n1=6,n2=20)
        kDatas["alpha191_83"] = alpha191_83(high_df,volume_df,n=5)
        kDatas["alpha191_84"] = alpha191_84(close_df,volume_df,n=20)
        kDatas["alpha191_85"] = alpha191_85(close_df,volume_df,n1=20,n2=8)
        kDatas["alpha191_86"] = alpha191_86(close_df,n1=10,n2=20)
        kDatas["alpha191_87"] = alpha191_87(open_df,high_df,low_df,vwap_df,n1=4,n2=7,n3=11)
        kDatas["alpha191_88"] = alpha191_88(close_df,n=20)
        kDatas["alpha191_89"] = alpha191_89(close_df,n1=13,n2=27,n3=10)
        kDatas["alpha191_90"] = alpha191_90(vwap_df, volume_df,n=5)
        kDatas["alpha191_91"] = alpha191_91(close_df,volume_df,low_df,n1=5,n2=40)
        kDatas["alpha191_92"] = alpha191_92(close_df,vwap_df,volume_df,n1=2,n2=3,n3=13,n4=5,n5=15)
        kDatas["alpha191_93"] = alpha191_93(open_df,low_df,n=20)
        kDatas["alpha191_94"] = alpha191_94(close_df,volume_df,n=30)
        kDatas["alpha191_95"] = alpha191_95(amount_df,n=20)
        kDatas["alpha191_96"] = alpha191_96(close_df,high_df, low_df, n1=9, n2=3)
        kDatas["alpha191_97"] = alpha191_97(volume_df,n=10)
        kDatas["alpha191_98"] = alpha191_98(close_df,n1=100,n2=100,n3=3)
        kDatas["alpha191_99"] = alpha191_99(close_df,volume_df,n=5)
        kDatas["alpha191_100"] = alpha191_100(volume_df,n=20)
        kDatas["alpha191_101"] = alpha191_101(close_df,volume_df,high_df,vwap_df,n1=30,n2=37,n3=15,n4=11)
        kDatas["alpha191_102"] = alpha191_102(volume_df,n=6)
        kDatas["alpha191_103"] = alpha191_103(low_df,n=20)
        kDatas["alpha191_104"] = alpha191_104(close_df,high_df,volume_df,n=5,m=20)
        kDatas["alpha191_105"] = alpha191_105(close_df,volume_df,n=10)
        kDatas["alpha191_106"] = alpha191_106(close_df,n=20)
        kDatas = kDatas.copy()
        kDatas["alpha191_107"] = alpha191_107(open_df,high_df,close_df,low_df)
        kDatas["alpha191_108"] = alpha191_108(high_df,vwap_df, volume_df,n=6,m=120)
        kDatas["alpha191_109"] = alpha191_109(high_df,low_df,n=10,m=2)
        kDatas["alpha191_110"] = alpha191_110(close_df,high_df,low_df,n=20)
        kDatas["alpha191_111"] = alpha191_111(close_df,low_df,high_df,volume_df,n=11,m=2)
        kDatas["alpha191_112"] = alpha191_112(close_df,n=12)
        kDatas["alpha191_113"] = alpha191_113(close_df,volume_df,n=5,m=20)
        kDatas["alpha191_114"] = alpha191_114(close_df,high_df,low_df,volume_df,vwap_df,n=5)
        kDatas["alpha191_115"] = alpha191_115(close_df,high_df,low_df,volume_df,n=30,m=10)
        kDatas["alpha191_116"] = alpha191_116(close_df,n=20)
        kDatas["alpha191_117"] = alpha191_117(close_df,high_df,low_df,volume_df,ret_df=None,n=32, m=16)
        kDatas["alpha191_118"] = alpha191_118(open_df, high_df, low_df,n=20)
        kDatas["alpha191_119"] = alpha191_119(open_df,vwap_df,volume_df,n=5,m=26)
        kDatas["alpha191_120"] = alpha191_120(close_df,vwap_df)
        kDatas["alpha191_121"] = alpha191_121(volume_df,vwap_df,n=20,m=60)
        #
        #####kDatas["alpha191_122"] = alpha191_122(close_df,n=13)#过于耗时，先去掉
        kDatas["alpha191_123"] = alpha191_123(high_df,low_df,volume_df,n=20,m=60)
        kDatas["alpha191_124"] = alpha191_124(close_df,vwap_df,n=30)
        kDatas["alpha191_125"] = alpha191_125(close_df,volume_df,vwap_df,n=17,m=20)
        kDatas["alpha191_126"] = alpha191_126(close_df,high_df,low_df)
        kDatas["alpha191_127"] = alpha191_127(close_df,n=12)
        kDatas["alpha191_128"] = alpha191_128(close_df,high_df,low_df,volume_df,n=14)
        kDatas["alpha191_129"] = alpha191_129(close_df,n = 12)
        kDatas["alpha191_130"] = alpha191_130(high_df, low_df, volume_df, vwap_df, n=9, m=10, k=7, l=3)
        kDatas["alpha191_131"] = alpha191_131(close_df,volume_df, vwap_df, n=18, m=18)
        kDatas["alpha191_132"] = alpha191_132(amount_df, n=20)
        kDatas["alpha191_133"] = alpha191_133(high_df,low_df,n=20)
        kDatas["alpha191_134"] = alpha191_134(close_df,volume_df,n=12)
        kDatas["alpha191_135"] = alpha191_135(close_df,n=20)
        kDatas["alpha191_136"] = alpha191_136(close_df,volume_df,n=10)
        kDatas["alpha191_137"] = alpha191_137(open_df,close_df,high_df,low_df)
        kDatas["alpha191_138"] = alpha191_138(close_df,volume_df,vwap_df,n=20)
        kDatas["alpha191_139"] = alpha191_139(open_df,volume_df,n=10)
        kDatas["alpha191_140"] = alpha191_140(open_df, close_df, high_df, low_df, volume_df, n=8)
        kDatas["alpha191_141"] = alpha191_141(high_df,volume_df,n=9)
        kDatas["alpha191_142"] = alpha191_142(close_df,volume_df,n1=10,n2=1,n3=5)
        kDatas["alpha191_143"] = alpha191_143(close_df)
        kDatas["alpha191_144"] = alpha191_144(close_df,amount_df,n=20)
        kDatas["alpha191_145"] = alpha191_145(volume_df,n1=9,n2=26,n3=12)
        kDatas["alpha191_146"] = alpha191_146(close_df,n1=20,n2=61,n3=60,m=2)
        kDatas["alpha191_147"] = alpha191_147(close_df,n=12)
        kDatas["alpha191_148"] = alpha191_148(open_df,volume_df,n1=6,n2=14)
        ####kDatas["alpha191_149"] = alpha191_149(close_df,benchmark_close_df,close_ret_df,benchmark_close_ret_df,n=252)
        kDatas["alpha191_150"] = alpha191_150(close_df,high_df,low_df,volume_df)
        kDatas["alpha191_151"] = alpha191_151(close_df,n=20)
        kDatas = kDatas.copy()
        kDatas["alpha191_152"] = alpha191_152(close_df,n=9,m=12,l=26)
        kDatas["alpha191_153"] = alpha191_153(close_df)
        kDatas["alpha191_154"] = alpha191_154(vwap_df,volume_df,n=180,m=16,l=18)
        kDatas["alpha191_155"] = alpha191_155(volume_df,n=13,m=27,l=10)
        kDatas["alpha191_156"] = alpha191_156(close_df,open_df,low_df,volume_df,vwap_df,n=5,m=2,l=3)
        kDatas["alpha191_157"] = alpha191_157(close_df,high_df,volume_df,vwap_df,n=30,m=37)
        kDatas["alpha191_158"] = alpha191_158(close_df,high_df, low_df,n=15, m=2)
        kDatas["alpha191_159"] = alpha191_159(close_df,high_df,low_df,n=6,m=12,l=24)
        kDatas["alpha191_160"] = alpha191_160(close_df,n=20,m=1)
        kDatas["alpha191_161"] = alpha191_161(close_df,high_df,low_df,n=12)
        kDatas["alpha191_162"] = alpha191_162(close_df,n=12,m=1)
        kDatas["alpha191_163"] = alpha191_163(close_df,volume_df,vwap_df,high_df,n=20)
        ###kDatas["alpha191_164"] = alpha191_164(close_df,high_df,low_df,l=12,n=13,m=2)
        kDatas["alpha191_165"] = alpha191_165(close_df,n=48)
        kDatas["alpha191_166"] = alpha191_166(close_df,n=20)
        kDatas["alpha191_167"] = alpha191_167(close_df,n=12)
        kDatas["alpha191_168"] = alpha191_168(volume_df,n=20)
        kDatas["alpha191_169"] = alpha191_169(close_df,n=9,m=12)
        kDatas["alpha191_170"] = alpha191_170(close_df,volume_df,high_df,vwap_df,n=5)
        kDatas["alpha191_171"] = alpha191_171(open_df,close_df,high_df,low_df,n=5)
        kDatas["alpha191_172"] = alpha191_172(close_df,high_df,low_df,n=14)
        #####kDatas["alpha191_173"] = alpha191_173(close_df,n=13)
        kDatas["alpha191_174"] = alpha191_174(close_df,n=20)
        kDatas["alpha191_175"] = alpha191_175(close_df,high_df,low_df,n=6)
        kDatas["alpha191_176"] = alpha191_176(close_df,high_df,low_df,volume_df,n=12, m=6)
        kDatas["alpha191_177"] = alpha191_177(high_df,n=20)
        kDatas["alpha191_178"] = alpha191_178(close_df,n=20)
        kDatas["alpha191_179"] = alpha191_179(volume_df, vwap_df, low_df,n=50, m=4, l=12)
        kDatas["alpha191_180"] = alpha191_180(close_df,volume_df,n=20,m=7)
        # # ######kDatas["alpha191_181"] = alpha191_181(close_df,benchmark_close_df,n=20)
        # # ######kDatas["alpha191_182"] = alpha191_182(close_df,open_df,banchmark_close_df,banchmark_open_df,n=20)
        kDatas["alpha191_183"] = alpha191_183(close_df,n=24)
        kDatas["alpha191_184"] = alpha191_184(close_df,open_df,n=200)
        kDatas["alpha191_185"] = alpha191_185(close_df,open_df)
        kDatas["alpha191_186"] = alpha191_186(close_df,high_df,low_df,n=14,m=6)
        kDatas["alpha191_187"] = alpha191_187(close_df,high_df,n=20)
        kDatas["alpha191_188"] = alpha191_188(high_df,low_df,n=11,m=2)
        kDatas["alpha191_189"] = alpha191_189(close_df,n=6)
        ######kDatas["alpha191_190"] = alpha191_190(close_df,n=20)
        kDatas["alpha191_191"] = alpha191_191(close_df,high_df,low_df,volume_df,n=20,m=5)


        # 去掉Nan值
        #kDatas = kDatas.dropna()
        return kDatas


if __name__ == '__main__':
    kDatas = pd.read_csv('./DataFiles/000016.csv')
    columnlist = ['nOpen', 'nHigh', 'nLow', 'nClose']
    for column in columnlist:
        kDatas[column] = kDatas[column]/10000.0

    tc = Technical()
    kDatas = tc.CalIndicator(kDatas)
    kDatas = tc.CalAlpaha191Factors(kDatas)
    kDatas.to_csv('./DataFiles/000016_indicator.csv',index=False)




