#二十三个元特征的名称，可以展示给用户
features_calculation = \
    ('目标属性类别个数', '目标属性类别的信息熵', '目标属性单个类别占的最大比例', '目标属性单个类别占的最小比例',
    'numeral 属性个数',' category 属性个数','numeral 属性所占比例','属性总数'
    'record 个数','category 属性中类别最少的属性 的类别个数','category 属性中类别最少的属性 的类别的 信息熵'
    'category 属性中类别最少的属性 的类别中 单个类别占的最大比例',' category 属性中类别最少的属性 的类别中 单个类别占的最小比例'
    'category 属性中类别最多的属性 的类别个数','category 属性中类别最多的属性 的类别的 信息熵',
     'category 属性中类别最多的属性 的类别中 单个类别占的最大比例'
    'category 属性中类别最多的属性 的类别中 单个类别占的最小比例','numeral 属性中最小的平均值',' numeral 属性中最大的平均值'
    ' numeral 属性中最小的方差','numeral 属性中最大的方差','numeral 属性的平均值的方差','numeral 属性的方差的方差')
features_init = \
    ['目标属性类别个数', '目标属性类别的信息熵', '目标属性单个类别占的最大比例', '目标属性单个类别占的最小比例',
    'numeral 属性个数',' category 属性个数','numeral 属性所占比例','属性总数'
    'record 个数','category 属性中类别最少的属性 的类别个数','category 属性中类别最少的属性 的类别的 信息熵'
    'category 属性中类别最少的属性 的类别中 单个类别占的最大比例',' category 属性中类别最少的属性 的类别中 单个类别占的最小比例'
    'category 属性中类别最多的属性 的类别个数','category 属性中类别最多的属性 的类别的 信息熵',
     'category 属性中类别最多的属性 的类别中 单个类别占的最大比例'
    'category 属性中类别最多的属性 的类别中 单个类别占的最小比例','numeral 属性中最小的平均值',' numeral 属性中最大的平均值'
    ' numeral 属性中最小的方差','numeral 属性中最大的方差','numeral 属性的平均值的方差','numeral 属性的方差的方差']

def featurelist_choose():
    feature_list = []
    feature_dict = {}
    for i in range(len(features_calculation)):
        feature_list.append(features_init.index(features_calculation[i]))
        feature_dict[i] = features_init.index(features_calculation[i])
    return feature_list,feature_dict

feature_list,feature_dict=featurelist_choose()
import pandas as pd
import os
import numpy as np
import math
from sklearn.preprocessing import StandardScaler

def calculate_features(featurelist,X,y,m,n,start,end,df,dfv):
    # 计算数据集特征
    # 特征一： 目标属性类别个数
    features=[]
    if 0 in featurelist:
        features.append(len(y.unique()))
    #
    tmp = list(y.value_counts())
    tmp.sort()
    # 特征二： 目标属性类别的信息熵
    if 1 in featurelist:
        entropy = 0
        for x in tmp:
            entropy = entropy + (-1 * x / m) * math.log(x / m, 2)
        features.append(entropy)
    # 特征三： 目标属性单个类别占的最大比例
    if 2 in featurelist:
        features.append(tmp[-1] / m)
    # 特征四： 目标属性单个类别占的最小比例
    if 3 in featurelist:
        features.append(tmp[0] / m)
    #
    tmp = list(df.dtypes)
    num_index, cate_index = list(), list()
    for i in range(start, end):
        if tmp[i] == 'float64' or tmp[i] == "int64":
            num_index.append(i - start)
        else:
            cate_index.append(i - start)
    # 特征五： numeral 属性个数
    if 4 in featurelist:
        features.append(len(num_index))
    # 特征六： category 属性个数
    if 5 in featurelist:
        features.append(len(cate_index))
    # 特征七： numeral 属性所占比例
    if 6 in featurelist:
        features.append(len(num_index) / (n - 1))
    # 特征八： 属性总数
    if 7 in featurelist:
        features.append(n - 1)
    # 特征九： record 个数
    if 8 in featurelist:
        features.append(m)
    #
    min_cls_ind, max_cls_ind = -1, -1
    if len(cate_index) != 0:
        min_cls_ind, max_cls_ind = 0, 0
    cate_cls_num = list()
    for i in cate_index:
        contents = pd.Series(np.array([dfv[j][i] for j in range(m)]))
        cate_cls_num.append(len(contents.unique()))
    for i in range(len(cate_index)):
        if cate_cls_num[min_cls_ind] > cate_cls_num[i]:
            min_cls_ind = i
        if cate_cls_num[max_cls_ind] < cate_cls_num[i]:
            max_cls_ind = i
    # 特征十： category 属性中类别最少的属性 的类别个数
    # 特征十一： category 属性中类别最少的属性 的类别的 信息熵
    # 特征十二： category 属性中类别最少的属性 的类别中 单个类别占的最大比例
    # 特征十三： category 属性中类别最少的属性 的类别中 单个类别占的最小比例
    if min_cls_ind != -1:
        if 9 in featurelist:
            features.append(cate_cls_num[min_cls_ind])
        contents = pd.Series(np.array([dfv[j][cate_index[min_cls_ind]] for j in range(m)]))
        tmp = list(contents.value_counts())
        tmp.sort()
        if 10 in featurelist:
            entropy = 0
            for x in tmp:
                entropy = entropy + (-1 * x / m) * math.log(x / m, 2)
            features.append(entropy)
        if 11 in featurelist:
            features.append(tmp[-1] / m)
        if 12 in featurelist:
            features.append(tmp[0] / m)
    else:
        if 9 in featurelist:
            features.append(0)
        if 10 in featurelist:
            features.append(0)
        if 11 in featurelist:
            features.append(0)
        if 12 in featurelist:
            features.append(0)
    #
    # 特征十四： category 属性中类别最多的属性 的类别个数
    # 特征十五： category 属性中类别最多的属性 的类别的 信息熵
    # 特征十六： category 属性中类别最多的属性 的类别中 单个类别占的最大比例
    # 特征十七： category 属性中类别最多的属性 的类别中 单个类别占的最小比例
    if max_cls_ind != -1:
        if 13 in featurelist:
            features.append(cate_cls_num[max_cls_ind])
        contents = pd.Series(np.array([dfv[j][cate_index[max_cls_ind]] for j in range(m)]))
        tmp = list(contents.value_counts())
        tmp.sort()
        if 14 in featurelist:
            entropy = 0
            for x in tmp:
                entropy = entropy + (-1 * x / m) * math.log(x / m, 2)
            features.append(entropy)
        if 15 in featurelist:
            features.append(tmp[-1] / m)
        if 16 in featurelist:
            features.append(tmp[0] / m)
    else:
        if 13 in featurelist:
            features.append(0)
        if 14 in featurelist:
            features.append(0)
        if 15 in featurelist:
            features.append(0)
        if 16 in featurelist:
            features.append(0)
    #
    # 特征十八： numeral 属性中最小的平均值
    # 特征十九： numeral 属性中最大的平均值
    # 特征二十： numeral 属性中最小的方差
    # 特征二一： numeral 属性中最大的方差
    # 特征二二： numeral 属性的平均值的方差
    # 特征二三： numeral 属性的方差的方差
    if len(num_index):
        NumX = np.array([[X[i][j] for j in num_index] for i in range(m)])
        scaler = StandardScaler()
        scaler.fit(NumX)
        NumX = scaler.transform(NumX)
        num_mean, num_var = list(), list()
        for i in range(len(num_index)):
            content = np.array([NumX[j][i] for j in range(m)])
            num_mean.append(content.mean())
            num_var.append(content.var())
        num_mean.sort()
        num_var.sort()
        if 17 in featurelist:
            features.append(num_mean[0])
        if 18 in featurelist:
            features.append(num_mean[-1])
        if 19 in featurelist:
            features.append(num_var[0])
        if 20 in featurelist:
            features.append(num_var[-1])
        if 21 in featurelist:
            features.append(np.var(num_mean))
        if 22 in featurelist:
            features.append(np.var(num_var))
    else:
        if 17 in featurelist:
            features.append(0)
        if 18 in featurelist:
            features.append(0)
        if 19 in featurelist:
            features.append(0)
        if 20 in featurelist:
            features.append(0)
        if 21 in featurelist:
            features.append(0)
        if 22 in featurelist:
            features.append(0)
    return features

