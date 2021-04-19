import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Evaluation
from weka.core.classes import Random
from weka.classifiers import Classifier
from sko.GA import GA
import time
import Details
import Evaluation

classifiers,classifier_dict,classifier_dict_EA=Details.details()

class AdaBoostM1():
    def __init__(self,filename):
        self.index = 1
        self.filename=filename
        self.classname="weka.classifiers.meta.AdaBoostM1"
        self.n_dim=3
        self.GA_size_pop=30
        self.max_iter=100
        self.lb = classifier_dict_EA[self.classname][1]
        self.ub = classifier_dict_EA[self.classname][2]
        self.precision = classifier_dict_EA[self.classname][4]
        self.cls_options=classifier_dict_EA[self.classname][0]
        self.default_options=classifier_dict_EA[self.classname][3]


    def func(self,*x):
        if self.index == 0:
            self.classp='start'
        else:
            self.classp = 'end'
        cls_options = []
        cls_options.append("-P")
        cls_options.append(str(int(x[0][0])))
        cls_options.append("-I")
        cls_options.append(str(int(x[0][1])))
        if x[0][2] > 0:
            cls_options.append("-Q")
        return -Evaluation.evaluation(self.filename,'AUC',self.classname,cls_options,self.classp)


    def putin(self,best_x):
        cls_options = []
        cls_options.append("-P")
        cls_options.append(str(int(best_x[0])))
        cls_options.append("-I")
        cls_options.append(str(int(best_x[1])))
        if best_x[2] > 0:
            cls_options.append("-Q")
        return cls_options

class BayesNet():
    def __init__(self,filename):
        self.index = 1
        self.filename = filename
        self.classname="weka.classifiers.bayes.BayesNet"
        self.n_dim=2
        self.GA_size_pop=30
        self.max_iter=100
        self.lb = classifier_dict_EA[self.classname][1]
        self.ub = classifier_dict_EA[self.classname][2]
        self.precision = classifier_dict_EA[self.classname][4]
        self.cls_options = classifier_dict_EA[self.classname][0]
        self.default_options = classifier_dict_EA[self.classname][3]

    def func(self,*x):
        if self.index == 0:
            self.classp = 'start'
        else:
            self.classp = 'end'
        cls_options = []
        if x[0][0] > 0:
            cls_options.append("-D")
        if len(cls_options) > 0:
            cls = Classifier(classname="weka.classifiers.bayes.BayesNet", options=cls_options)
        else:
            cls = Classifier(classname="weka.classifiers.bayes.BayesNet")
        return -Evaluation.evaluation(self.filename, 'AUC', self.classname, cls_options, self.classp)
    def putin(self,best_x):
        cls_options = []
        if best_x[0] > 0:
            cls_options.append("-D")
        return cls_options

class Bagging():
    def __init__(self,filename):
        self.index = 1
        self.filename = filename
        self.classname="weka.classifiers.meta.Bagging"
        self.n_dim=3
        self.GA_size_pop=30
        self.max_iter=100
        self.lb = classifier_dict_EA[self.classname][1]
        self.ub = classifier_dict_EA[self.classname][2]
        self.precision = classifier_dict_EA[self.classname][4]
        self.cls_options = classifier_dict_EA[self.classname][0]
        self.default_options = classifier_dict_EA[self.classname][3]

    def func(self,*x):
        if self.index == 0:
            self.classp = 'start'
        else:
            self.classp = 'end'
        cls_options = []
        cls_options.append("-I")
        cls_options.append(str(int(x[0][0])))
        cls_options.append("-P")
        cls_options.append(str(int(x[0][1])))
        if x[0][2] > 0:
            cls_options.append("-O")

        return -Evaluation.evaluation(self.filename,'AUC',self.classname,cls_options,self.classp)

    def putin(self,best_x):
        cls_options = []
        cls_options.append("-I")
        cls_options.append(str(int(best_x[0])))
        cls_options.append("-P")
        cls_options.append(str(int(best_x[1])))
        if best_x[2] > 0:
            cls_options.append("-O")
        return cls_options
class ClassificationViaRegression():
    def __init__(self,filename):
        self.index = 1
        self.filename = filename
        self.classname="weka.classifiers.meta.ClassificationViaRegression"
        self.n_dim=1
        self.GA_size_pop=30
        self.max_iter=50
        self.lb = classifier_dict_EA[self.classname][1]
        self.ub = classifier_dict_EA[self.classname][2]
        self.precision = classifier_dict_EA[self.classname][4]
        self.cls_options = classifier_dict_EA[self.classname][0]
        self.default_options = classifier_dict_EA[self.classname][3]

    def func(self,*x):
        if self.index == 0:
            self.classp = 'start'
        else:
            self.classp = 'end'
        cls_options = []
        if x[0] > 0:
            cls_options.append('--')
            cls_options.append('-M')
            cls_options.append(str(int(x[0])))
        return -Evaluation.evaluation(self.filename, 'AUC', self.classname, cls_options, self.classp)
    def putin(self,best_x):
        cls_options = []
        if best_x[0] > 0:
            cls_options.append("--")
            cls_options.append('-M')
            cls_options.append(str(int(best_x[0])))
        return cls_options
class DecisionTable():
    def __init__(self,filename):
        self.index = 1
        self.filename = filename
        self.classname="weka.classifiers.rules.DecisionTable"
        self.n_dim=1
        self.GA_size_pop=30
        self.max_iter=100
        self.lb = classifier_dict_EA[self.classname][1]
        self.ub = classifier_dict_EA[self.classname][2]
        self.precision = classifier_dict_EA[self.classname][4]
        self.cls_options = classifier_dict_EA[self.classname][0]
        self.default_options = classifier_dict_EA[self.classname][3]

    def func(self,*x):
        if self.index == 0:
            self.classp = 'start'
        else:
            self.classp = 'end'
        cls_options = []
        cls_options.append("-X")
        cls_options.append(str(int(x[0][0])))
        # cls_options.append("-S")
        # cls_options.append(str(int(x[1])))

        cls = Classifier(classname="weka.classifiers.rules.DecisionTable", options=cls_options)
        train, test = self.data.train_test_split(80.0, Random(1))
        cls.build_classifier(train)
        evl = Evaluation(train)
        evl.test_model(cls, test)

        return -evl.percent_correct / 100 * evl.weighted_area_under_roc

    def putin(self, best_x):
        cls_options = []
        cls_options.append("-X")
        cls_options.append(str(int(best_x[0])))
        return cls_options
class IBK():
    def __init__(self,filename):
        self.index = 1
        self.filename = filename
        self.classname="weka.classifiers.lazy.IBk"
        self.n_dim=4
        self.GA_size_pop=30
        self.max_iter=100
        self.lb = classifier_dict_EA[self.classname][1]
        self.ub = classifier_dict_EA[self.classname][2]
        self.precision = classifier_dict_EA[self.classname][4]
        self.cls_options = classifier_dict_EA[self.classname][0]
        self.default_options = classifier_dict_EA[self.classname][3]

    def func(self,*x):
        if self.index == 0:
            self.classp = 'start'
        else:
            self.classp = 'end'
        cls_options = []
        if x[0][0] > 0:
            cls_options.append("-E")
        if x[0][1] > 0:
            cls_options.append("-X")
        if x[0][2] > 0:
            cls_options.append("-I")
        cls_options.append("-K")
        cls_options.append(str(int(x[0][3])))
        return -Evaluation.evaluation(self.filename,'AUC',self.classname,cls_options,self.classp)
    def putin(self, best_x):
        cls_options = []
        if best_x[0] > 0:
            cls_options.append("-E")
        if best_x[1] > 0:
            cls_options.append("-X")
        if best_x[2] > 0:
            cls_options.append("-I")
        cls_options.append("-K")
        cls_options.append(str(int(best_x[3])))
        return cls_options
class JRip():
    def __init__(self,filename):
        self.index = 1
        self.filename = filename
        self.classname = "weka.classifiers.rules.JRip"
        self.n_dim = 4
        self.GA_size_pop = 30
        self.max_iter = 100
        self.lb = classifier_dict_EA[self.classname][1]
        self.ub = classifier_dict_EA[self.classname][2]
        self.precision = classifier_dict_EA[self.classname][4]
        self.cls_options = classifier_dict_EA[self.classname][0]
        self.default_options = classifier_dict_EA[self.classname][3]

    def func(self, *x):
        if self.index == 0:
            self.classp = 'start'
        else:
            self.classp = 'end'
        cls_options = []
        if x[0][0] > 0:
            cls_options.append("-F")
        if x[0][1] > 0:
            cls_options.append("-N")
        if x[0][2] > 0:
            cls_options.append("-O")
        cls_options.append("-S")
        cls_options.append(str(int(x[0][3])))
        return -Evaluation.evaluation(self.filename, 'AUC', self.classname, cls_options, self.classp)

    def putin(self, best_x):
        cls_options = []
        cls_options.append("-F")
        cls_options.append(str(int(best_x[0])))
        cls_options.append("-N")
        cls_options.append(str(int(best_x[1])))
        cls_options.append("-O")
        cls_options.append(str(int(best_x[2])))
        cls_options.append("-S")
        cls_options.append(str(int(best_x[3])))
        return cls_options

class KStar():
    def __init__(self,filename):
        self.index = 1
        self.filename = filename
        self.classname="weka.classifiers.lazy.KStar"
        self.n_dim=2
        self.GA_size_pop=20
        self.max_iter=200
        self.lb = classifier_dict_EA[self.classname][1]
        self.ub = classifier_dict_EA[self.classname][2]
        self.precision = classifier_dict_EA[self.classname][4]
        self.cls_options = classifier_dict_EA[self.classname][0]
        self.default_options = classifier_dict_EA[self.classname][3]

    def func(self,*x):
        if self.index == 0:
            self.classp = 'start'
        else:
            self.classp = 'end'
        cls_options = []
        cls_options.append("-M")
        if -1 < x[0][0] <= -0.5:
            cls_options.append("a")
        elif -0.5 < x[0][0] <= 0:
            cls_options.append("d")
        elif 0 < x[0][0] <= 0.5:
            cls_options.append("m")
        else:
            cls_options.append("n")
        cls_options.append("-B")
        cls_options.append(str(int(x[0][1])))
        return -Evaluation.evaluation(self.filename,'AUC',self.classname,cls_options,self.classp)
    def putin(self, best_x):
        cls_options = []
        cls_options.append("-M")
        if -1 < best_x[0] <= -0.5:
            cls_options.append("a")
        elif -0.5 < best_x[0] <= 0:
            cls_options.append("d")
        elif 0 < best_x[0] <= 0.5:
            cls_options.append("m")
        else:
            cls_options.append("n")
        cls_options.append("-B")
        cls_options.append(str(int(best_x[1])))
        return cls_options
class LMT():
    def __init__(self,filename):
        self.index = 1
        self.filename = filename
        self.classname = "weka.classifiers.trees.LMT"
        self.n_dim = 3
        self.GA_size_pop = 30
        self.max_iter = 100
        self.lb = classifier_dict_EA[self.classname][1]
        self.ub = classifier_dict_EA[self.classname][2]
        self.precision = classifier_dict_EA[self.classname][4]
        self.cls_options = classifier_dict_EA[self.classname][0]
        self.default_options = classifier_dict_EA[self.classname][3]

    def func(self, *x):
        if self.index == 0:
            self.classp = 'start'
        else:
            self.classp = 'end'
        cls_options = []
        cls_options.append("-I")
        cls_options.append(str(int(x[0][0])))
        cls_options.append("-M")
        cls_options.append(str(int(x[0][1])))
        cls_options.append("-W")
        cls_options.append(str(x[0][2]))

        return -Evaluation.evaluation(self.filename,'AUC',self.classname,cls_options,self.classp)

    def putin(self, best_x):
        cls_options = []
        cls_options.append("-I")
        cls_options.append(str(int(best_x[0])))
        cls_options.append("-M")
        cls_options.append(str(int(best_x[1])))
        cls_options.append("-W")
        cls_options.append(str(best_x[2]))
        return cls_options


class Logistic():
    def __init__(self,filename):
        self.index = 1
        self.filename = filename
        self.classname="weka.classifiers.functions.Logistic"
        self.n_dim=1
        self.GA_size_pop=30
        self.max_iter=100
        self.lb = classifier_dict_EA[self.classname][1]
        self.ub = classifier_dict_EA[self.classname][2]
        self.precision = classifier_dict_EA[self.classname][4]
        self.cls_options = classifier_dict_EA[self.classname][0]
        self.default_options = classifier_dict_EA[self.classname][3]

    def func(self,*x):
        if self.index == 0:
            self.classp = 'start'
        else:
            self.classp = 'end'
        cls_options = []
        cls_options.append("-R")
        cls_options.append(str(x[0][0]))
        cls = Classifier(classname="weka.classifiers.functions.Logistic", options=cls_options)

        return -Evaluation.evaluation(self.filename,'AUC',self.classname,cls_options,self.classp)

    def putin(self, best_x):
        cls_options = []
        cls_options.append("-R")
        cls_options.append(str(best_x[0]))
        return cls_options

class LogitBoost():
    def __init__(self,filename):
        self.index = 1
        self.filename = filename
        self.classname="weka.classifiers.meta.LogitBoost"
        self.n_dim=3
        self.GA_size_pop=30
        self.max_iter=100
        self.lb = classifier_dict_EA[self.classname][1]
        self.ub = classifier_dict_EA[self.classname][2]
        self.precision = classifier_dict_EA[self.classname][4]
        self.cls_options = classifier_dict_EA[self.classname][0]
        self.default_options = classifier_dict_EA[self.classname][3]

    def func(self,*x):
        if self.index == 0:
            self.classp = 'start'
        else:
            self.classp = 'end'
        cls_options = []
        cls_options.append("-O")
        cls_options.append(str(int(x[0][0])))
        cls_options.append("-P")
        cls_options.append(str(int(x[0][1])))
        cls_options.append("-H")
        cls_options.append(str(x[0][2]))

        return -Evaluation.evaluation(self.filename,'AUC',self.classname,cls_options,self.classp)

    def putin(self, best_x):
        cls_options = []
        cls_options.append("-O")
        cls_options.append(str(int(best_x[0])))
        cls_options.append("-P")
        cls_options.append(str(int(best_x[1])))
        cls_options.append("-H")
        cls_options.append(str(int(best_x[2])))
        return cls_options

class MultilayerPerceptron():
    def __init__(self,filename):
        self.index = 1
        self.filename = filename
        self.classname="weka.classifiers.functions.MultilayerPerceptron"
        self.n_dim=5
        self.GA_size_pop=50
        self.max_iter=100
        self.lb = classifier_dict_EA[self.classname][1]
        self.ub = classifier_dict_EA[self.classname][2]
        self.precision = classifier_dict_EA[self.classname][4]
        self.cls_options = classifier_dict_EA[self.classname][0]
        self.default_options = classifier_dict_EA[self.classname][3]

    def func(self,*x):
        if self.index == 0:
            self.classp = 'start'
        else:
            self.classp = 'end'
        cls_options = []
        cls_options.append("-L")
        cls_options.append(str(x[0][0]))
        cls_options.append("-M")
        cls_options.append(str(x[0][1]))
        if x[0][2] > 0:
            cls_options.append("-B")
        if x[0][3] > 0:
            cls_options.append("-C")
        if x[0][4] > 0:
            cls_options.append("-D")

        return -Evaluation.evaluation(self.filename,'AUC',self.classname,cls_options,self.classp)

    def putin(self, best_x):
        cls_options = []
        cls_options.append("-L")
        cls_options.append(str(int(best_x[0])))
        cls_options.append("-M")
        cls_options.append(str(int(best_x[1])))
        if best_x[2] > 0:
            cls_options.append("-B")
        if best_x[3] > 0:
            cls_options.append("-C")
        if best_x[4] > 0:
            cls_options.append("-D")
        return cls_options
class NaiveBayes():
    def __init__(self,filename):
        self.index = 1
        self.filename = filename
        self.classname="weka.classifiers.bayes.NaiveBayes"
        self.n_dim=2
        self.GA_size_pop=30
        self.max_iter=100
        self.lb = classifier_dict_EA[self.classname][1]
        self.ub = classifier_dict_EA[self.classname][2]
        self.precision = classifier_dict_EA[self.classname][4]
        self.cls_options = classifier_dict_EA[self.classname][0]
        self.default_options = classifier_dict_EA[self.classname][3]

    def func(self,*x):
        if self.index == 0:
            self.classp = 'start'
        else:
            self.classp = 'end'
        cls_options = []
        if x[0][0]> 0 and x[0][1] < 0:
            cls_options.append("-D")
        if x[0][0] > 0 and x[0][1] < 0:
            cls_options.append("-K")
        if len(cls_options) > 0:
            cls = Classifier(classname="weka.classifiers.bayes.NaiveBayes", options=cls_options)
        else:
            cls = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
        return -Evaluation.evaluation(self.filename,'AUC',self.classname,cls_options,self.classp)

    def putin(self, best_x):
        cls_options = []
        if best_x[0] > 0:
            cls_options.append("-D")
        if best_x[1] > 0:
            cls_options.append("-K")
        return cls_options

class RandomCommittee():
    def __init__(self,filename):
        self.index = 1
        self.filename = filename
        self.classname="weka.classifiers.meta.RandomCommittee"
        self.n_dim=1
        self.GA_size_pop=20
        self.max_iter=100
        self.lb = classifier_dict_EA[self.classname][1]
        self.ub = classifier_dict_EA[self.classname][2]
        self.precision = classifier_dict_EA[self.classname][4]
        self.cls_options = classifier_dict_EA[self.classname][0]
        self.default_options = classifier_dict_EA[self.classname][3]

    def func(self,*x):
        if self.index == 0:
            self.classp = 'start'
        else:
            self.classp = 'end'
        cls_options = []
        cls_options.append("-I")
        cls_options.append(str(int(x[0][0])))

        return -Evaluation.evaluation(self.filename, 'AUC', self.classname, cls_options, self.classp)
    def putin(self, best_x):
        cls_options = []
        cls_options.append("-I")
        cls_options.append(str(int(best_x[0])))
        return cls_options
class RandomForest():
    def __init__(self,filename):
        self.index = 1
        self.filename = filename
        self.classname="weka.classifiers.trees.RandomForest"
        self.n_dim=3
        self.GA_size_pop=30
        self.max_iter=40
        self.lb = classifier_dict_EA[self.classname][1]
        self.ub = classifier_dict_EA[self.classname][2]
        self.precision = classifier_dict_EA[self.classname][4]
        self.cls_options = classifier_dict_EA[self.classname][0]
        self.default_options = classifier_dict_EA[self.classname][3]

    def func(self,*x):
        if self.index == 0:
            self.classp = 'start'
        else:
            self.classp = 'end'
        cls_options = []
        cls_options.append("-I")
        cls_options.append(str(int(x[0][0])))
        cls_options.append("-depth")
        cls_options.append(str(int(x[0][1])))
        cls_options.append("-K")
        cls_options.append(str(int(x[0][2])))

        return -Evaluation.evaluation(self.filename,'AUC',self.classname,cls_options,self.classp)
    def putin(self, best_x):
        cls_options = []
        cls_options.append("-I")
        cls_options.append(str(int(best_x[0])))
        cls_options.append("-depth")
        cls_options.append(str(int(best_x[1])))
        cls_options.append("-K")
        cls_options.append(str(int(best_x[2])))
        return cls_options
class RandomSubSpace():
    def __init__(self,filename):
        self.index = 1
        self.filename = filename
        self.classname="weka.classifiers.meta.RandomSubSpace"
        self.n_dim=2
        self.GA_size_pop=30
        self.max_iter=50
        self.lb = classifier_dict_EA[self.classname][1]
        self.ub = classifier_dict_EA[self.classname][2]
        self.precision = classifier_dict_EA[self.classname][4]
        self.cls_options = classifier_dict_EA[self.classname][0]
        self.default_options = classifier_dict_EA[self.classname][3]
    def func(self,*x):
        if self.index == 0:
            self.classp = 'start'
        else:
            self.classp = 'end'
        cls_options = []
        cls_options.append("-I")
        cls_options.append(str(int(x[0][0])))
        cls_options.append("-P")
        cls_options.append(str(x[0][1]))

        return -Evaluation.evaluation(self.filename,'AUC',self.classname,cls_options,self.classp)
    def putin(self, best_x):
        cls_options = []
        cls_options.append("-I")
        cls_options.append(str(int(best_x[0])))
        cls_options.append("-P")
        cls_options.append(str(best_x[1]))
        return cls_options
class Vote():
    def __init__(self,filename):
        self.index = 1
        self.filename = filename
        self.classname="weka.classifiers.meta.Vote"
        self.n_dim=1
        self.GA_size_pop=30
        self.max_iter=100
        self.lb = classifier_dict_EA[self.classname][1]
        self.ub = classifier_dict_EA[self.classname][2]
        self.precision = classifier_dict_EA[self.classname][4]
        self.cls_options = classifier_dict_EA[self.classname][0]
        self.default_options = classifier_dict_EA[self.classname][3]

    def func(self,*x):
        if self.index == 0:
            self.classp = 'start'
        else:
            self.classp = 'end'
        cls_options = []
        cls_options.append("-S")
        cls_options.append(str(int(x[0][0])))

        cls = Classifier(classname="weka.classifiers.meta.Vote", options=cls_options)
        train, test = self.data.train_test_split(75.0, Random(1))
        cls.build_classifier(train)
        evl = Evaluation(train)
        evl.test_model(cls, test)
        return -evl.percent_correct / 100
    def putin(self, best_x):
        cls_options = []
        cls_options.append("-S")
        cls_options.append(str(best_x[0]))
        return cls_options


def hpo(HPO_name,filename,hpo_iteration_num):
    '''

    :param HPO_name: 算法名称
    :param filename: 数据集路径
    :param hpo_iteration_num: 超参数优化最大代数
    :return:
    '''
    HPO_list_name = []
    for classifier in classifiers:
        HPO_list_name.append(classifier.split('.')[3])

    HPO_list_act = [AdaBoostM1, BayesNet, Bagging, ClassificationViaRegression, DecisionTable,
                IBK, KStar, Logistic, LogitBoost, MultilayerPerceptron, NaiveBayes, RandomCommittee,
                RandomForest, RandomSubSpace, Vote,JRip,LMT]
    if HPO_name in HPO_list_name:
        HPO_index = HPO_list_name.index(HPO_name)
        HPO_act = HPO_list_act[HPO_index](filename)

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
        options_type=classifier_dict[HPO_act.classname][1]
        i=0
        for type in options_type:
            if type=='list':
                options[2*i+1]=str(classifier_dict[HPO_act.classname][3][i])
            i=i+1

        print(HPO_act.classname +str(options))
        if options==[]:
            classname=HPO_act.classname
            cls=Classifier(classname=classname)
            options=cls.options

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
