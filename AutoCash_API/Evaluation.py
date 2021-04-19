from weka.core.converters import Loader
from weka.classifiers import Evaluation
from weka.classifiers import Classifier
from weka.core.classes import Random
from weka.classifiers import FilteredClassifier
import time
from weka.filters import Filter
import math


def evaluation(filename, evaluation_indicator, classifier, option, classp):
    '''

    :param filename:数据集名字
    :param evaluation_indicator: AUC/ACC/……
    :param classifier: 算法名称
    :param option: 超参数值列表
    :param classp: 标志数据集分类信息是在最前一行或是最后一行(start/end)
    :return:
    '''

    schema = []  # TODO 使用SQL查询指定表的schema，也就是每一列的名称
    if classp == 'start':
        label_name = schema[0]
    else:
        label_name = schema[-1]
    sql_1 = f'select SQL(select count(distinct {label_name}) from {filename}) as n_class\n'\
            f'select SQL(select * from {filename}) as raw_data\n' \
            f'create tensor split_ratio(2,) from ones((2, ))\n' \
            f'select split_ratio[0]*0.7 as split_ratio[0]\n' \
            f'select split_ratio[1]*0.3 as split_ratio[1]\n' \
            f'select SplitDataset(raw_data, split_ratio) as split_res\n' \
            f'select split_res[0] as train_data\n' \
            f'select split_res[1] as test_data\n'
    if classp == 'start':
        sql_2 = f'select train_data[:,0] as train_y\n' \
                f'select train_data[:,1] as train_x\n'
    else:
        sql_2 = f'select train_data[:,-1] as train_y\n' \
                f'select train_data[:,-2] as train_x\n'
    sql_1 += sql_2
    if classifier == 'KNN':
        # TODO: KNN API
        sql_2 = f'select KNN()'
    elif classifier == 'logistic':
        iter_times = option['M']
        sql_2 = f'create tensor lr(1,) from 0.01\n' \
                f'create tensor threshold(1,) from 0.3\n' \
                f'create tensor iter_times(1,) from {iter_times}\n' \
                f'select logistic(train_x, train_y, n_class, lr, threshold, iter_times)\n'

    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(filename)
    if classp == "start":
        data.class_is_first()
    elif classp == "end":
        data.class_is_last()

    remove = Filter(classname="weka.filters.unsupervised.attribute.Remove")
    if option == []:
        option = Classifier(classname=classifier).options
    cls = Classifier(classname=classifier, options=option)

    time1 = time.time()
    cls.build_classifier(data)
    time2 = time.time()

    fc = FilteredClassifier()
    fc.filter = remove
    fc.classifier = cls

    evl = Evaluation(data)
    evl.crossvalidate_model(fc, data, 10, Random(1))

    ACC = evl.percent_correct / 100
    all_time = time2 - time1
    F1_score = evl.f_measure(0)
    precison = evl.precision(0)
    recall = evl.recall(0)
    if math.isnan(evl.weighted_area_under_roc):
        AUC = 0
    else:
        AUC = evl.percent_correct / (evl.weighted_area_under_roc * 100)

    evl_dict = {'ACC': ACC, 'time': all_time, 'F1-score': F1_score, 'precison': precison, 'recall': recall, 'AUC': AUC}

    return evl_dict[evaluation_indicator]
