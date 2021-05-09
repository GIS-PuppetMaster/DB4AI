from sko.GA import GA
import time
import Details
import Evaluation

classifiers,classifier_dict,classifier_dict_EA=Details.details()

class ClassAlgorithm():
    def __init__(self, algname, filename, evaluation_indicator, classp):
        self.filename=filename
        self.algname=algname
        self.evaluation_indicator=evaluation_indicator
        self.classp=classp
        self.n_dim=len(classifier_dict_EA[self.algname][1])
        self.GA_size_pop=30
        self.max_iter=100
        self.lb = classifier_dict_EA[self.algname][1]
        self.ub = classifier_dict_EA[self.algname][2]
        self.precision = classifier_dict_EA[self.algname][4]
        self.cls_options=classifier_dict_EA[self.algname][0]
        self.default_options=classifier_dict_EA[self.algname][3]

    def func(self,*x):
        cls_options = []
        for i in range(len(self.cls_options)):
            cls_options.append(self.cls_options[i])
            cls_options.append(x[0][i])
        return -Evaluation.evaluation(self.filename,self.evaluation_indicator,self.algname,cls_options,self.classp)

    def putin(self,best_x):
        cls_options = []
        for i in range(len(self.cls_options)):
            cls_options.append(self.cls_options[i])
            cls_options.append(best_x[i])
        return cls_options

def hpo(HPO_name,filename,hpo_iteration_num,evaluation_indicator,classp):
    '''

    :param HPO_name: 算法名称
    :param filename: 数据集路径
    :param hpo_iteration_num: 超参数优化最大代数
    :return:
    '''
    HPO_list_name = []
    for classifier in classifiers:
        HPO_list_name.append(classifier)

    HPO_list_act = [ClassAlgorithm(algname,filename,evaluation_indicator,classp) for algname in HPO_list_name]
    if HPO_name in HPO_list_name:
        HPO_index = HPO_list_name.index(HPO_name)
        HPO_act = HPO_list_act[HPO_index]

        a = time.time()
        if hpo_iteration_num<HPO_act.max_iter:
            HPO_act.max_iter=hpo_iteration_num

        ga = GA(HPO_act.func, n_dim=HPO_act.n_dim, size_pop=HPO_act.GA_size_pop, max_iter=HPO_act.max_iter,
                lb=HPO_act.lb, ub=HPO_act.ub, precision=HPO_act.precision)

        best_x, best_y = ga.run()
        b = time.time()

        print("time: " + str(b - a))
        print("best performance: " + str(-best_y) + "\n")

        options=HPO_act.putin(best_x)
        #把list类型的超参数映射回去
        options_type=classifier_dict[HPO_act.algname][1]
        i=0
        for type in options_type:
            if type=='list':
                list_options = eval(classifier_dict[HPO_act.algname][2][i].split(','))
                options[2*i+1] = list_options[int(options[2*i+1])]
            i=i+1

        if options == []:
            options = classifier_dict[HPO_act.algname][3]

        import pandas as pd
        import matplotlib.pyplot as plt

        inverse = [-x for x in ga.all_history_Y]
        Y_history = pd.DataFrame(inverse)

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        Y_history.mean(axis=1).plot(kind='line')
        plt.show()
    else:
        print("不存在这种算法")
    return options

'''
# a simple test
HPO_name,filename,hpo_iteration_num,evaluation_indicator,classp = "weka.classifiers.meta.AdaBoostM1", "", 50, "AUC", "end"
hpo(HPO_name,filename,hpo_iteration_num,evaluation_indicator,classp)
'''
