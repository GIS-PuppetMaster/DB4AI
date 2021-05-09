import os
import Details
import time
import Evaluation
import math
##分配算法
def Allocationalgorithm(time_limit,evaluation_indicator):
    '''

    :param time_limit: 本部分最长时间限制
    :param evaluation_indicator: 评估指标(AUC/ACC/……)
    :return:
    '''
    time1=time.time()

    classifiers, full_name_of_classifiers,hpo_dict = Details.details()

    first_data_dir = "./Arff/first_total"
    last_data_dir = "./Arff/last_total"

    # all data sets file name
    first_data_list = os.listdir(first_data_dir)
    last_data_list = os.listdir(last_data_dir)

    for filename in first_data_list:
        print("这是数据集 " + filename)
        filename = first_data_dir + '/' + filename
        flag=Allocation(time1, time_limit, filename, 'start', full_name_of_classifiers, evaluation_indicator)
        if flag==1:
            return

    for filename in last_data_list:
        print("这是数据集 " + filename)
        filename = last_data_dir + '/' + filename
        flag = Allocation(time1, time_limit, filename, 'end', full_name_of_classifiers, evaluation_indicator)
        if flag == 1:
            return


def Allocation(pre_time,time_limit,filename,classp,full_name_of_classifiers,evaluation_indicator):
    best_classifier_dict = {}
    classifier_dict = {}
    for classifier in full_name_of_classifiers:
        print("这是算法 " + classifier)
        sum = 0.0
        try:
            for i in range(5):
                print("这是第 " + str(i) + " 次训练：")
                option = []
                if evaluation_indicator == 'time':
                    sum -= Evaluation.evaluation(filename, evaluation_indicator, classifier, option, classp)
                else:
                    sum += Evaluation.evaluation(filename, evaluation_indicator, classifier, option, classp)
                time2 = time.time()
                time_used = time2 - pre_time
                if time_used > time_limit:
                    if len(classifier_dict) > 0:
                        best_classifier_dict[filename] = \
                            sorted(classifier_dict.items(), key=lambda item: item[1], reverse=True)[0][0]
                        with open("Customization_result.txt", "a+") as f:
                            f.write(filename + '\t' + best_classifier_dict[filename] + '\n')
                    classifier_dict.clear()
                    return 1

        except:
            print("有异常，" + classifier + " 跳过")
        else:
            classifier_dict[classifier] = sum / 5

    if len(classifier_dict) > 0:
        best_classifier_dict[filename] = sorted(classifier_dict.items(), key=lambda item: item[1], reverse=True)[0][0]
        with open("Customization_result.txt", "a+") as f:
            f.write(filename + '\t' + best_classifier_dict[filename] + '\n')
    classifier_dict.clear()
    return 0