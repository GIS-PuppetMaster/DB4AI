from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np


def try_clfs(X_train,FeatureList,data_file):
    evaluation_dict={}
    featureLength = len(FeatureList)
    featurelist = []
    for i in range(featureLength):
        featurelist.append(i)

    X_data = np.loadtxt(data_file, delimiter=',', usecols=tuple(featurelist))
    Y_data = np.loadtxt(data_file, delimiter=',', usecols=featureLength, dtype=str)


    clf = RandomForestClassifier()
    t0 = time.time()
    clf.fit(X_data, Y_data)

    score = clf.score(X_data, Y_data)
    t1 = time.time()
    print(clf.predict(X_train))

    print(score)
    t2 = time.time()

    print("The fit time of RF: " + str(t1 - t0))
    print("The time for predicting new datasets: " + str(t2 - t1))

    evaluation_dict["score"]=score
    evaluation_dict["The time for predicting new datasets"]=t2 - t1
    return clf.predict(X_train)

##接口，训练随机森林，推荐算法
def modelFit(X_train,FeatureList,data_file):
    print("The clf is: " + 'random_forest')
    predict=try_clfs(X_train,FeatureList,data_file)
    return predict
