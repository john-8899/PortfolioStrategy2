# -*- coding:utf-8 -*-
import pandas as pd
from collections import defaultdict
from deepShap import PortfolioStrategy
class CFeatureSelection:
    def __init__(self):
        # 技术指标因子
        self.technicalFactor = ['DIF', 'DEA', 'MACD', 'MA5', 'MA10', 'MA20', 'MA60', 'MA120', 'MA250', 'BOLLUP', 'BOLL',
                                'BOLLDN','DEMA5', 'DEMA10', 'DEMA20', 'DEMA60', 'DEMA120', 'DEMA250', 'EMA5', 'EMA10', 'EMA20',
                                'EMA50','EMA200', 'HT_TRENDLINE', 'KAMA10', 'KAMA20', 'KAMA30', 'MAMA', 'FAMA', 'MAVP',
                                'MIDPOINT','MIDPRICE', 'SAR', 'T3', 'TRIMA20', 'TRIMA50', 'TRIMA200', 'WMA20', 'WMA50', 'WMA200',
                                'ADX5','ADX14', 'ADX21', 'ADXR', 'APO', 'AROONDOWN', 'AROONUP', 'AROONOSC', 'BOP', 'CCI',
                                'CMO', 'DX','MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP',
                                'ROCR', 'ROCR100','RSI6', 'RSI12', 'RSI24', 'SLOWK', 'SLOWD', 'FASTK', 'FASTD', 'fastkRSI', 'fastdRSI',
                                'TRIX','ULTOSC', 'WILLR', 'AD', 'ADOSC', 'OBV', 'HT_DCPERIOD', 'HT_DCPHASE', 'INPHASE',
                                'QUADRATURE','HT_SINE', 'HT_TRENDMODE', 'ATR', 'NATR', 'TRANGE']
        # alpha191 因子
        self.Alpha191Factors = [f'alpha191_{i + 1}' for i in range(191) if
                                i + 1 not in [30, 75, 122, 149, 164, 173, 181, 182, 190]]
        print("原始特征总数:", len(self.technicalFactor) + len(self.Alpha191Factors))

    def dfs(self,node,visited,components, graph):
        """
            # DFS查找连通分量（即相关特征组）
        :param node: 当前节点
        :param visited: 已访问过的节点
        :param components 组
        :return:
        """
        visited.add(node)
        components.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                self.dfs(neighbor, visited, components, graph)

    def feature_corr_analysis(self,featuresList,corrThreshold=0.9):
        """
        基于相关性分析特征，将高度相关的特征分组。
        该方法读取股票数据，计算特征间的Pearson相关系数矩阵，
        根据设定的阈值筛选高度相关的特征，并通过图算法找出所有连通分量（特征组）。

        步骤：
        1. 读取股票数据文件
        2. 计算所有特征间的绝对Pearson相关系数矩阵
        3. 构建特征关系图（相关性超过阈值的特征互相连接）
        4. 使用深度优先搜索(DFS)找出所有连通分量（特征组）
        5. 输出分组结果和统计信息

        :return: None
        """
        datas = pd.read_csv("./DataFiles/kDatas/000016.csv")

        # 计算Pearson相关系数矩阵并取绝对值（只关心相关性强度，不关心方向）
        corrMatrix = datas[featuresList].corr(method='pearson').abs()

        # 设定相关性阈值，超过该阈值的特征将被视为高度相关
        corrThreshold = 0.9

        # 使用邻接表表示特征关系图
        graph = defaultdict(list)

        # 遍历上三角矩阵（避免重复计算和自相关）
        for i in range(len(featuresList)):
            for j in range(i + 1, len(featuresList)):
                if corrMatrix.iloc[i, j] >= corrThreshold:
                    feat_i = featuresList[i]
                    feat_j = featuresList[j]
                    # 在图中添加双向边（无向图）
                    graph[feat_i].append(feat_j)
                    graph[feat_j].append(feat_i)

        # 使用DFS算法找出所有连通分量（特征组）
        visited = set()
        feature_groups = []
        for node in featuresList:
            if node not in visited:
                components = []
                self.dfs(node, visited, components, graph)
                feature_groups.append(components)

        # 输出分组结果和统计信息
        for i, group in enumerate(feature_groups):
            print(f"特征组{i + 1}: {group}")
        print(f"分组前特征数量：{len(featuresList)}，特征组总数: {len(feature_groups)}")

        return feature_groups


    def feature_selection_correlation(self):
        """
        基于相关性进行特征选择，主要步骤包括：
        1. 将特征按相关性分组
        2. 使用SHAP值评估每组特征的重要性
        3. 筛选每组中SHAP值最高的特征
        4. 对筛选后的特征再次训练并排序
        :return:
        """
        # 获取待分析的特征列表
        featureList = self.technicalFactor + ['nOpen', 'nHigh', 'nLow', 'nClose', 'iVolume']  # + self.Alpha191Factors + ['nOpen', 'nHigh', 'nLow', 'nClose', 'iVolume']
        print("原始特征总数:", len(featureList))
        # 基于相关性把特征分组
        feature_groups = self.feature_corr_analysis(featureList)

        # 训练模型, 并使用SHAP值进行特征重要性评估
        cfs = PortfolioStrategy()
        dict_shap_values = cfs.myMain_train_shap(featureList)

        #筛选每组中的shap 值最高的特征
        selectedFeatures = []
        for i, group in enumerate(feature_groups):
            newData = {}
            for feature in group:
                if feature in dict_shap_values:
                    newData[feature] = dict_shap_values[feature]
                else:
                    print(f"特征 {feature} 不在字典中")
            #按新的V降序排序
            newDict = dict(sorted(newData.items(), key=lambda item: item[1], reverse=True))
            #获取newDict中的第一个值，即shap值最大的特征
            firstFeature = next(iter(newDict.keys()))
            selectedFeatures.append(firstFeature)

        print(f"相关性分组筛选特征数：{len(selectedFeatures)} 筛选特征: {selectedFeatures}")

        #根据筛选后的特征训练模型，然后对筛选后的特征按shap 值进行排序
        shapData = cfs.myMain_train_shap(selectedFeatures)
        # 按shap 值降序排序
        shapData = dict(sorted(shapData.items(), key=lambda item: item[1], reverse=True))
        print(f"特征排序(Dict)：{len(shapData)}: {shapData}")
        print(f"特征排序(list)：{list(shapData.keys())}")
        return list(shapData.keys())

    def select_features_front_n(self,featureList):
        """
        基于前N个特征进行特征选择
        基于feature_selection_correlation函数筛选的特征，并按shap排序后
        计算前N个特征的预测值，看是否超过全部特征，尽量筛选少的特征取得好的结果
        :return:
        """
        fps = PortfolioStrategy()
        resultList = []
        for n in range(20,len(featureList)):
            tmpfeatureList = featureList[:n]
            acccuracy,f1,c1precision,c1recall,c1f1 = fps.myMain_train(tmpfeatureList)
            print(f"{n},准确率:{acccuracy:.4f},F1:{f1:.4f},C1-precision:{c1precision:.4f},C1-recall:{c1recall:.4f},C1-F1:{c1f1:.4f}")
            resultList.append([n,acccuracy,f1,c1precision,c1recall,c1f1])
        pdData = pd.DataFrame(resultList,columns=['n','accuracy','f1','c1precision','c1recall','c1f1'])
        pdData.to_csv("./DataFiles/featureSelection.csv",index=False)

if __name__ == '__main__':
    fs = CFeatureSelection()
    selectfeatrues = fs.feature_selection_correlation()
    fs.select_features_front_n(selectfeatrues)