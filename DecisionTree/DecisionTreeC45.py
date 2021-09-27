from DecisionTreeID3 import DecisionTreeID3
from collections import namedtuple
from PlotTree import PlotTree
from math import log2
import pandas as pd
import numpy as np


class DecisionTreeC45(DecisionTreeID3):
    '''
    C4.5决策树
    '''

    InfoGainRatioCateFeature = namedtuple('InfoGainRatioFeature',
                                          ['type', 'feature', 'info_gain_ratio', 'info_gain'])

    InfoGainRatioNumFeature = namedtuple('InfoGainRatioFeature',
                                         ['type', 'feature', 'feature_value', 'info_gain_ratio', 'info_gain'])

    def __init__(self, *args, **kwargs):
        self.feature_rate_field = 'feature_rate'
        super(DecisionTreeC45, self).__init__(*args, **kwargs)

    # 分割数据 支持空数据
    def _split_df(self, data_df, field, field_value, add_na=False):
        df_field_df = data_df[data_df[field] == field_value]
        if add_na:
            df_null_df = data_df[data_df[field].isnull()]
            df_null_df[self.feature_rate_field] = df_null_df[self.feature_rate_field] * df_field_df.shape[0] / (
                    data_df.shape[0] - df_null_df.shape[0])
            df_field_df = pd.concat([df_field_df, df_null_df])

        return df_field_df.drop(field, axis=1)

    # 计算信息熵
    def _get_entropy(self, data_df, label):
        data_num = data_df.shape[0]
        entropy_value = -data_df.groupby(label).sum()[self.feature_rate_field].apply(
            lambda x: x / data_num * log2(x / data_num)).sum()

        return entropy_value

    # 获取数值型的信息增益及信息增益率
    def _get_num_feature_entropy(self, data_df, feature):
        info_gain_ratio_list = []
        data_num = data_df.shape[0]
        data_not_null_df = data_df[data_df[feature].notnull()]
        data_not_null_num = data_not_null_df.shape[0]
        base_entropy = self._get_entropy(data_not_null_df, self.label)
        num_feature_value_list = data_not_null_df[feature].sort_values().unique().tolist()
        feature_value_list = [round((num_feature_value_list[i] + num_feature_value_list[i + 1]) / 2, 3) for i in
                              range(len(num_feature_value_list) - 1)]

        for feature_value in feature_value_list:
            lt_df = data_not_null_df[data_not_null_df[feature] <= feature_value]
            gt_df = data_not_null_df[data_not_null_df[feature] > feature_value]
            lt_prob = lt_df.shape[0] / data_not_null_df.shape[0]
            gt_prob = gt_df.shape[0] / data_not_null_df.shape[0]
            new_entropy = lt_prob * self._get_entropy(lt_df,
                                                      self.label) + gt_prob * self._get_entropy(gt_df,
                                                                                                self.label)

            info_gain = round((base_entropy - new_entropy) * (data_not_null_num / data_num), 4)
            iv = -(lt_prob * log2(lt_prob) + gt_prob * log2(gt_prob))
            info_gain_ratio = round(info_gain / iv, 4)
            info_gain_ratio_list.append(
                self.InfoGainRatioNumFeature('num', feature, feature_value, info_gain_ratio, info_gain))

        return info_gain_ratio_list

    # 获取类别型的信息增益及信息增益率
    def _get_cate_feature_entropy(self, data_df, feature):

        data_not_null_df = data_df[data_df[feature].notnull()]
        base_entropy = self._get_entropy(data_not_null_df, self.label)
        base_entropy_rate = data_not_null_df[self.feature_rate_field].sum() / data_df[self.feature_rate_field].sum()
        new_entropy = 0
        for feature_value in data_not_null_df[feature].unique():
            sub_data_df = self._split_df(data_df, feature, feature_value)
            prob = sub_data_df[self.feature_rate_field].sum() / data_not_null_df[self.feature_rate_field].sum()
            new_entropy += prob * self._get_entropy(sub_data_df, self.label)
        iv = self._get_entropy(data_not_null_df, feature)
        info_gain = round(base_entropy_rate * (base_entropy - new_entropy), 4)
        info_gain_ratio = round(info_gain / iv, 4)
        return self.InfoGainRatioCateFeature('cate', feature, info_gain_ratio, info_gain)

    # 选择最优特征作为划分点
    def choose_best_feature(self, data_df, feature_list):
        info_gain_feature_list = []
        for feature in feature_list:
            if data_df[feature].dtypes in ('float', 'int'):
                info_gain_feature_list.extend(self._get_num_feature_entropy(data_df, feature))
            else:
                info_gain_feature_list.append(self._get_cate_feature_entropy(data_df, feature))

        info_gain_avg = sum([i.info_gain for i in info_gain_feature_list]) / len(info_gain_feature_list)
        return sorted([i for i in info_gain_feature_list if i.info_gain >= info_gain_avg], key=lambda x: x.info_gain_ratio,
                      reverse=True)[0]

    # 创建决策树
    def create(self, data_df=None, feature_list=None):

        if data_df is None:
            self.data_df[self.feature_rate_field] = 1
            data_df = self.data_df

        feature_list = data_df.drop(self.label, axis=1).columns.tolist() if feature_list is None else feature_list
        feature_list = [feature for feature in feature_list if feature != self.feature_rate_field]

        if len(data_df[self.label].unique()) == 1:
            label = data_df[self.label].unique()[0]
            return label

        # 没有特征可继续分隔时 若存在不同的label时 取出现次数最多的label为最终label
        if not feature_list:
            label = data_df[self.label].value_counts(ascending=False).index[0]
            return label

        best_feature_object = self.choose_best_feature(data_df, feature_list)
        best_feature_type = best_feature_object.type
        best_feature = best_feature_object.feature
        this_tree = {best_feature: {}}
        if best_feature_type == 'cate':
            filter_feature_list = [feature for feature in feature_list if feature != best_feature]
            for best_feature_value in data_df[data_df[best_feature].notnull()][best_feature].unique():
                this_tree[best_feature][best_feature_value] = {}
                sub_data_df = self._split_df(data_df, best_feature, best_feature_value, add_na=True)
                this_tree[best_feature][best_feature_value] = self.create(sub_data_df, filter_feature_list)

        else:
            best_feature_value = best_feature_object.feature_value
            lt_df = data_df[data_df[best_feature] <= best_feature_value]
            this_tree[best_feature][f'<={best_feature_value}'] = self.create(lt_df, feature_list)
            gt_df = data_df[data_df[best_feature] > best_feature_value]
            this_tree[best_feature][f'>{best_feature_value}'] = self.create(gt_df, feature_list)

        return this_tree


if __name__ == '__main__':

    df = pd.DataFrame([[np.nan, '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, '是'],
                       ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', np.nan, 0.774, '是'],
                       ['乌黑', '蜷缩', np.nan, '清晰', '凹陷', '硬滑', 0.634, '是'],
                       ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, '是'],
                       [np.nan, '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, '是'],
                       ['青绿', '稍蜷', '浊响', '清晰', np.nan, '软粘', 0.403, '是'],
                       ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, '是'],
                       ['乌黑', '稍蜷', '浊响', np.nan, '稍凹', '硬滑', 0.437, '是'],
                       ['乌黑', np.nan, '沉闷', '稍糊', '稍凹', '硬滑', 0.666, '否'],
                       ['青绿', '硬挺', '清脆', np.nan, '平坦', '软粘', 0.243, '否'],
                       ['浅白', '硬挺', '清脆', '模糊', '平坦', np.nan, 0.245, '否'],
                       ['浅白', '蜷缩', np.nan, '模糊', '平坦', '软粘', 0.343, '否'],
                       [np.nan, '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, '否'],
                       ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, '否'],
                       ['乌黑', '稍蜷', '浊响', '清晰', np.nan, '软粘', 0.36, '否'],
                       ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, '否'],
                       ['青绿', np.nan, '沉闷', '稍糊', '稍凹', '硬滑', 0.719, '否']],
                      columns=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '好瓜(标签)'])

    # df = pd.DataFrame([[np.nan, '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
    #                    ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', np.nan, '是'],
    #                    ['乌黑', '蜷缩', np.nan, '清晰', '凹陷', '硬滑', '是'],
    #                    ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
    #                    [np.nan, '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
    #                    ['青绿', '稍蜷', '浊响', '清晰', np.nan, '软粘', '是'],
    #                    ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],
    #                    ['乌黑', '稍蜷', '浊响', np.nan, '稍凹', '硬滑', '是'],
    #                    ['乌黑', np.nan, '沉闷', '稍糊', '稍凹', '硬滑', '否'],
    #                    ['青绿', '硬挺', '清脆', np.nan, '平坦', '软粘', '否'],
    #                    ['浅白', '硬挺', '清脆', '模糊', '平坦', np.nan, '否'],
    #                    ['浅白', '蜷缩', np.nan, '模糊', '平坦', '软粘', '否'],
    #                    [np.nan, '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
    #                    ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
    #                    ['乌黑', '稍蜷', '浊响', '清晰', np.nan, '软粘', '否'],
    #                    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
    #                    ['青绿', np.nan, '沉闷', '稍糊', '稍凹', '硬滑', '否']],
    #                   columns=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜(标签)'])

    tree = DecisionTreeC45(df, '好瓜(标签)')
    PlotTree.plot(tree.tree)
