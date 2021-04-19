import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd

def loadUserData(filename,classp):
    Draw(filename,classp)

def Draw(filename,classp):
    filepath=filename
    data = arff.loadarff(filepath)
    df = pd.DataFrame(data[0])
    if classp=="start":
        labels = df.values[:, 0]  # 要处理的标签
    else:
        labels=df.values[:, -1]
    cla = []  # 处理后的标签
    num = []
    for label in labels:
        if label.decode('gbk') not in cla:
            cla.append(label.decode('gbk'))
    cla_len = len(cla)
    for i in range(cla_len):
        num.append(0)
    for label in labels:
        num[cla.index(label.decode('gbk'))] = num[cla.index(label.decode('gbk'))] + 1

    labels = cla
    X = num

    plt.pie(X, labels=labels, autopct='%1.2f%%')  # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
    plt.legend(loc='upper left', bbox_to_anchor=(-0.1, 1))
    plt.title("Classification of data sets")
    plt.savefig("Pie.png")

    plt.show()


