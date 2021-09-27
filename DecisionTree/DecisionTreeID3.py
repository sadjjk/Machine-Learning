from collections import namedtuple
from PlotTree import PlotTree
import pandas as pd
from math import log2


# ID3决策树
class DecisionTreeID3:
    '''
    ID3决策树
    '''

    InfoGainFeature = namedtuple('InfoGainFeature', ['feature', 'info_gain', 'feature_entropy', 'base_entropy'])

    def __init__(self, data_df, label):
        '''
        初始化
        :param data_df: 数据源 类型 pandas DataFrame
        :param label: 标签字段名
        '''
        self.data_df = data_df
        self.label = label
        self.tree = self.create()

    # 计算信息熵
    def _get_entropy(self, data_df, label):
        data_num = data_df.shape[0]
        entropy_value = -data_df[label].value_counts().apply(lambda x: x / data_num * log2(x / data_num)).sum()
        return entropy_value

    # 分割数据
    def _split_df(self, data_df, field, field_value):
        return data_df[data_df[field] == field_value].drop(field, axis=1)

    # 选择最优特征作为划分点
    def choose_best_feature(self, data_df, feature_list):
        base_entropy = self._get_entropy(data_df, self.label)
        info_gain_feature_list = []
        for feature in feature_list:
            new_entropy = 0.0
            for feature_value in data_df[feature].unique():
                sub_data_df = self._split_df(data_df, feature, feature_value)
                prob = sub_data_df.shape[0] / data_df.shape[0]
                new_entropy += prob * self._get_entropy(sub_data_df, self.label)
            info_gain_feature_list.append(
                self.InfoGainFeature(feature, base_entropy - new_entropy, new_entropy, base_entropy))
        return sorted(info_gain_feature_list, key=lambda x: x.info_gain, reverse=True)[0].feature

    # 创建决策树
    def create(self, data_df=None, feature_list=None):
        data_df = self.data_df if data_df is None else data_df
        feature_list = data_df.drop(self.label, axis=1).columns.tolist() if feature_list is None else feature_list

        if len(data_df[self.label].unique()) == 1:
            label = data_df[self.label].unique()[0]
            return label

        # 没有特征可继续分隔时 若存在不同的label时 取出现次数最多的label为最终label
        if not feature_list:
            label = data_df[self.label].value_counts(ascending=False).index[0]
            return label

        best_feature = self.choose_best_feature(data_df, feature_list)
        filter_feature_list = [feature for feature in feature_list if feature != best_feature]
        this_tree = {best_feature: {}}
        for best_feature_value in data_df[best_feature].unique():
            this_tree[best_feature][best_feature_value] = {}

            sub_data_df = self._split_df(data_df, best_feature, best_feature_value)
            this_tree[best_feature][best_feature_value] = self.create(sub_data_df, filter_feature_list)

        return this_tree




if __name__ == '__main__':

    df = pd.DataFrame([['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
                       ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
                       ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
                       ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
                       ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
                       ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
                       ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
                       ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
                       ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
                       ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
                       ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
                       ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
                       ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
                       ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
                       ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
                       ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
                       ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']], columns=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '类别'])

    tree = DecisionTreeID3(df, '类别')
    PlotTree.plot(tree.tree)
