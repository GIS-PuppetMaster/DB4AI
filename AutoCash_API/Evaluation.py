import time
from Executor import Executor
from SecondLevelLanguageParser import Parser


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
        column_name = schema[0]
    else:
        column_name = schema[-1]
    sql = f'select SQL(select count(*) from {filename} group by {column_name}) as n_classes\n' \
          f'select SQL(select * from {filename}) as raw_data\n' \
          f'create tensor split_ratio(2,) from ones((2, ))\n' \
          f'select split_ratio[0]*0.7 as split_ratio[0]\n' \
          f'select split_ratio[1]*0.3 as split_ratio[1]\n' \
          f'select SplitDataset(raw_data, split_ratio) as split_res\n' \
          f'select split_res[0] as train_data\n' \
          f'select split_res[1] as test_data\n' \
          f'create tensor mse(1,)\n' \
          f'create tensor auc(1,)\n' \
          f'create tensor f1(1,)\n' \
          f'create tensor acc(1,)\n' \
          f'create tensor recall(1,)\n' \
          f'create tensor prec(1,)\n'
    if classp == 'start':
        sql_2 = f'select train_data[:,0] as train_y\n' \
                f'select train_data[:,1] as train_x\n'
    else:
        sql_2 = f'select train_data[:,-1] as train_y\n' \
                f'select train_data[:,-2] as train_x\n'
    sql += sql_2
    if classifier == 'IBK':
        k = option['K']
        sql += f'create tensor k(1,) from {k}\n'
        if option['I']:
            sql += f'select KNN_I(acc,auc,prec,recall,mse,f1,test_x, test_y, train_x, train_y, {k})\n'
        elif option['F']:
            sql += f'select KNN_F(acc,auc,prec,recall,mse,f1, test_x, test_y, train_x, train_y, {k})\n'
        else:
            sql += f'select KNN(acc,auc,prec,recall,mse,f1, test_x, test_y, train_x, train_y, {k})\n'
    elif classifier == 'Logistic':
        iter_times = option['M']
        ridge = option['R']
        sql_2 = f'create tensor lr(1,) from 0.01\n' \
                f'create tensor ridge(1,) from {ridge}\n' \
                f'create tensor iter_times(1,) from {iter_times}\n' \
                f'select logistic(acc,auc,prec,recall,mse,f1, train_x, train_y, lr, n_classes, iter_times)\n'
        sql += sql_2
    elif classifier == 'SVM':
        iter_times = option['M']
        c = option['C']
        eps = option['E']
        # TODO test
        sql_2 = f'create tensor c(1, ) from {c}\n' \
                f'create tensor eps(1, ) from {eps}\n' \
                f'create tensor iter_times(1, ) from {iter_times}\n' \
                f'select SVM(acc,auc,prec,recall,mse,f1, x, y, c, eps, iter_times)\n'
        sql += sql_2
    elif classifier == 'RBF':
        iter_times = option['M']
        n_centers = option['N']
        batch_size = option['B']
        learning_rate = option['L']
        sql_2 = f'create tensor n_centers(1, ) from {n_centers}\n' \
                f'create tensor batch_size(1, ) from {batch_size}\n' \
                f'create tensor learning_rate(1, ) from {learning_rate}\n' \
                f'create tensor iter_times(1, ) from {iter_times}\n' \
                f'select SVM(acc,auc,prec,recall,mse,f1, test_x, test_y, train_x, train_y, n_centers, n_classes, learning_rate, iter_times)\n'
        sql += sql_2
    else:
        raise Exception(f'not supported algorithm:{classifier}')
    # elif classifier == 'LogitBoost':
    #     iter_times = option['M']
    #     sql_2 = f'create tensor iter_times(1,) from {iter_times}\n' \
    #             f'select logit_boost(train_x, train_y, n_class,{iter_times})'
    #     sql_1 += sql_2
    parser = Parser(sql)
    result = parser()
    executor = Executor(result)
    time1 = time.time()
    executor.run()
    time1 = time.time() - time1
    # auc = executor.var_dict['auc']
    # acc = executor.var_dict['acc']
    # recall = executor.var_dict['recall']
    # precision = executor.var_dict['prec']
    # mse = executor.var_dict['mse']
    # f1 = executor.var_dict['f1']

    # loader = Loader(classname="weka.core.converters.ArffLoader")
    # data = loader.load_file(filename)
    # if classp == "start":
    #     data.class_is_first()
    # elif classp == "end":
    #     data.class_is_last()
    #
    # remove = Filter(classname="weka.filters.unsupervised.attribute.Remove")
    # if option == []:
    #     option = Classifier(classname=classifier).options
    # cls = Classifier(classname=classifier, options=option)
    #
    # time1 = time.time()
    # cls.build_classifier(data)
    # time2 = time.time()
    #
    # fc = FilteredClassifier()
    # fc.filter = remove
    # fc.classifier = cls
    #
    # evl = Evaluation(data)
    #     evl.crossvalidate_model(fc, data, 10, Random(1))
    #
    # ACC = evl.percent_correct / 100
    # all_time = time2 - time1
    # F1_score = evl.f_measure(0)
    # precison = evl.precision(0)
    # recall = evl.recall(0)
    # if math.isnan(evl.weighted_area_under_roc):
    #     AUC = 0
    # else:
    #     AUC = evl.percent_correct / (evl.weighted_area_under_roc * 100)

    evl_dict = {'ACC': acc, 'time': time1, 'F1-score': f1, 'precison': precision, 'recall': recall, 'AUC': auc, 'MSE': mse}

    return evl_dict[evaluation_indicator]
