#计算用户输入的数据时使用
import pandas as pd
from scipy.io import arff
import numpy as np
import Features as fe
import os
#训练的时候使用
def getIndex():
    first_data_dir = "./Arff/first_total"
    last_data_dir = "./Arff/last_total"
    first_data_list = os.listdir(first_data_dir)
    last_data_list = os.listdir(last_data_dir)
    classIndex = {}
    for file in first_data_list:
        classIndex[file] = "start"
    for file in last_data_list:
        classIndex[file] = "end"
    return classIndex

#重新训练的时候使用
def calculate(featurelist, filename, classIndex):
    '''
        :param filename:
        :param featurelist: list，是选取的元特征的index，如[1,2,3]
        :param classIndex:
        :return:
        '''
    fileName = "./Arff/total" + "/" + filename
    data = arff.loadarff(fileName)
    df = pd.DataFrame(data[0])
    (m, n) = df.shape
    classp = classIndex[filename]
    if classp == "end":
        start, end, classindex = 0, n - 1, n
    elif classp == "start":
        start, end, classindex = 1, n, 0
    dfv = df.values
    X = np.array([[dfv[i][j] for j in range(start, end)] for i in range(m)])
    y = pd.Series(np.array([dfv[i][classindex] for i in range(m)]))
    features=fe.calculate_features(featurelist,X, y, m, n, start, end, df, dfv)
    return features


def loadData(filename,featurelist,classp):
    '''

    :param filename:
    :param featurelist: list，是选取的元特征的index
    :param classp:
    :return:
    '''
    data = arff.loadarff(filename)
    df = pd.DataFrame(data[0])
    (m, n) = df.shape
    dfv = df.values
    start = 0
    end = 0
    if classp == "end":
        start, end, classindex = 0, n - 1, n
    elif classp == "start":
        start, end, classindex = 1, n, 0
    X = np.array([[dfv[i][j] for j in range(start, end)] for i in range(m)])
    y = pd.Series(np.array([dfv[i][0] for i in range(m)]))
    features = fe.calculate_features(featurelist,X,y,m,n,start,end,df,dfv)
    return features