import loadUserData as lud
import HPO
import dqn
import weka.core.jvm as jvm
import Evaluation as ev
import GenerateDataSet as gds
import GenerateData as gd
import CaculateFeatures as cf
import MyModel as mm

def ClassSelectionMain(filename='pasture.arff', alg_selection_time=200,hpo_iteration_num=100,evaluation_indicator='AUC', classp='end', train_flag='n'):
    '''
    :param filename:数据集路径
    :param alg_selection_time:程序最大运行时间(s)
    :param hpo_iteration_num:hpo中遗传算法优化代数
    :param evaluation_indicator:AUC/ACC/……
    :param classp:标志数据集分类信息是在最前一行或是最后一行(start/end)
    :param train_flag:是否重新进行dqn训练(y/n)
    :return:算法名,超参数及取值,evaluation_indicator的值
    注意：如果要重新训练的话请多给一点时间，如果时间过少Customization_result.txt中只有一行数
    据程序将会报错
    '''
    with open("dataset1.txt", 'r+') as file:
        file.truncate(0)
    with open("Customization_result.txt", 'r+') as file:
        file.truncate(0)

    ##读取用户数据集进行分析
    FeatureList = [0, 2, 4, 6, 7, 9, 13]  # 选择的元特征的值

    features = []

    lud.loadUserData(filename, classp)

    if train_flag == "y":
        dqn_time_limit = alg_selection_time * 0.5
        FeatureList = dqn.dqn_train(dqn_time_limit)
        dataset = 'dataset1.txt'
        allocation_algorithm_time_limit= alg_selection_time * 0.5
        gd.Allocationalgorithm(allocation_algorithm_time_limit,evaluation_indicator)
        gds.GenerateDataSetaction(FeatureList)
    else:
        FeatureList = FeatureList
        dataset = 'dataset.txt'
    features.append(cf.loadData(filename, FeatureList, classp))
    alg = mm.modelFit(features, FeatureList, dataset)
    alg_name = alg[0].split('.')[3]
    option = HPO.hpo(alg_name, filename,hpo_iteration_num)
    evaluation=ev.evaluation(filename,evaluation_indicator,alg[0],option,classp)
    return alg_name,option,evaluation

# 不训练版本
# jvm.start()
# filename='pasture.arff'
# time=60
# evaluation_indicator='AUC'
# classp='end'
# train_flag='n'
# hpo_iteration_num=100
# print(ClassSelectionMain(filename,time,hpo_iteration_num,evaluation_indicator,classp,train_flag))
# jvm.stop()

# 训练版本
# jvm.start()
# filename='pasture.arff'
# time=2000
# evaluation_indicator='AUC'
# classp='end'
# train_flag='y'
# hpo_iteration_num=100
# print(ClassSelectionMain(filename,time,hpo_iteration_num,evaluation_indicator,classp,train_flag))
# jvm.stop()

