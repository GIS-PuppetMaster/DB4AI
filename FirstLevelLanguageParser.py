import re
from time import time

from Executor import Executor
from SecondLevelLanguageParser import Parser


def get_train_sql_head(data_name, label_pos):
    sql = f"select TensorFromSql({data_name}) as data\n"
    if label_pos == "head":
        sql += "select data[:,1:] as x\n" \
               "select data[:,0] as y\n"
    elif label_pos == "tail":
        sql += "create tensor temp_index(1,) from -1\n" \
               "select data[:,:-1] as x\n" \
               "select data[:,-1] as y\n"
    else:
        raise Exception("label_pos error")
    sql += "select shape(x) as sx\n" \
           "select 0.8 * sx[0] as train_len\n" \
           "select x[0:train_len,:] as train_x\n" \
           "select x[train_len:,:] as test_x\n" \
           "select y[0:train_len,:] as train_y\n" \
           "select y[train_len:,:] as test_y\n"
    return sql


def get_test_sql_head(data_name, label_pos):
    sql = f"select TensorFromSql({data_name}) as data\n"
    if label_pos == "head":
        sql += "select data[:,1:] as x\n" \
               "select data[:,0] as y\n"
    elif label_pos == "tail":
        sql += "create tensor temp_index(1,) from -1\n" \
               "select data[:,:-1] as x\n" \
               "select data[:,-1] as y\n"
    else:
        raise Exception("label_pos error")
    return sql


if __name__ == '__main__':

    # select AutoClassify(alg_selection_time:float, hpo_iteration_num:int, evaluation_indicator:[], classp, train_flag:bool) [as result_filename] from filename
    # select train(result_filename, classp, evaluation_indicator) from dataset
    # first_sql = input('please input ML training task\n')
    # first_sql = 'select AutoClassify(60, 10, \'acc\', \'end\', \'false\') from filename'
    # first_sql = "select train(\'logistic\', \'res_table\', \'tail\', \'acc\') as model from real_data;"
    first_sql = "select test(\'logistic\', \'model\', \'tail\', \'acc\') from test_data;"
    # parse
    filename_reg = '[a-zA-Z]+[a-zA-Z0-9_]*'
    # auto_classify_reg = f'^(select|SELECT)[ \t]+AutoClassify([(].*[)])[ \t]+((as|AS)[ \t]+({filename_reg})[ \t]+)?((from|FROM)[ \t]+({filename_reg}))'
    train_match = f'^(select|SELECT)[ \t]+train([(].*[)])[ \t]+((as|AS)[ \t]+({filename_reg})[ \t]+)?(from|FROM)[ \t]*([ \t]*{filename_reg})[ \t]*;'
    test_match = f'^(select|SELECT)[ \t]+test([(].*[)])[ \t]+(from|FROM)[ \t]+([ \t]*{filename_reg}[ \t]*);'
    # auto_match = re.match(auto_classify_reg, first_sql)
    train_match = re.match(train_match, first_sql)
    test_match = re.match(test_match, first_sql)
    print(train_match)
    if train_match:
        groups = train_match.groups()
        parameters = eval(groups[1])
        model_name = groups[4]  # can be None
        data_name = groups[6]
        algorithm = parameters[0]
        result_table = parameters[1]
        label_pos = parameters[2]
        metrics = parameters[3]
        if algorithm == 'logistic':
            sql = get_train_sql_head(data_name, label_pos)
            sql += "create tensor lr(1,) from 0.01\n" \
                   "create tensor class_num(1,) from 2\n" \
                   "create tensor ridge(1,) from 0.05\n" \
                   "create tensor iter_times(1,) from 50\n" \
                   "create tensor mse(1,)\n" \
                   "create tensor auc(1,)\n" \
                   "create tensor f1(1,)\n" \
                   "create tensor acc(1,)\n" \
                   "create tensor recall(1,)\n" \
                   "create tensor prec(1,)\n" \
                   "select logistic(acc,auc,prec,recall,mse,f1, test_x,test_y,x,y, ridge, lr, class_num, iter_times)"
        elif algorithm == 'Softmax':
            sql = get_train_sql_head(data_name, label_pos)
            sql += "create tensor lr(1,) from 0.01\n" \
                   "create tensor class_num(1,) from 3\n" \
                   "create tensor ridge(1,) from 0.01\n" \
                   "create tensor iter_times(1,) from 500\n" \
                   "create tensor mse(1,)\n" \
                   "create tensor auc(1,)\n" \
                   "create tensor f1(1,)\n" \
                   "create tensor acc(1,)\n" \
                   "create tensor recall(1,)\n" \
                   "create tensor prec(1,)\n" \
                   "select softmax_classification(acc,auc,prec,recall,mse,f1, test_x,test_y,x,y,class_num, ridge, lr, iter_times)"
        else:
            raise Exception("unsupported algorithm")
        path = f'operators/{algorithm}.sql'
        with open(path, 'r', encoding='utf-8') as f:
            create_test = f.readlines()
        Parser(create_test)()
        with open(path, 'r', encoding='utf-8') as f:
            create_test = f.readlines()
        sql = sql.split("\n")
        for i in range(len(sql)):
            sql[i] = sql[i] + "\n"
        result = Parser(sql)()
        executor = Executor(result)
        executor.run()
    elif test_match:
        groups = test_match.groups()
        parameters = eval(groups[1])
        model_name = groups[4]  # can be None
        data_name = groups[3]
        algorithm = parameters[0]
        model = parameters[1]
        label_pos = parameters[2]
        metrics = parameters[3]
        if algorithm == 'logistic':
            sql = get_test_sql_head(data_name, label_pos)
            sql += "create tensor class_num(1,) from 2\n" \
                   "create tensor mse(1,)\n" \
                   "create tensor auc(1,)\n" \
                   "create tensor f1(1,)\n" \
                   "create tensor acc(1,)\n" \
                   "create tensor recall(1,)\n" \
                   "create tensor prec(1,)\n" \
                   "select test_logistic(acc,auc,prec,recall,mse,f1, test_x,test_y, class_num)"
        elif algorithm == 'softmax':
            sql = get_test_sql_head(data_name, label_pos)
            sql += "create tensor class_num(1,) from 3\n" \
                   "create tensor mse(1,)\n" \
                   "create tensor auc(1,)\n" \
                   "create tensor f1(1,)\n" \
                   "create tensor acc(1,)\n" \
                   "create tensor recall(1,)\n" \
                   "create tensor prec(1,)\n" \
                   "select test_softmax_classification(acc,auc,prec,recall,mse,f1, test_x,test_y,class_num)"
        else:
            raise Exception("unsupported algorithm")
        path = f'operators/test_{algorithm}.sql'
        with open(path, 'r', encoding='utf-8') as f:
            create_test = f.readlines()
        Parser(create_test)()
        with open(path, 'r', encoding='utf-8') as f:
            create_test = f.readlines()
        sql = sql.split("\n")
        for i in range(len(sql)):
            sql[i] = sql[i] + "\n"
        result = Parser(sql)()
        executor = Executor(result)
        executor.run()
    else:
        raise Exception("error sql")
    # if auto_match:
    #     groups = auto_match.groups()
    #     parameters = eval(groups[1])
    #     alg_selection_time = float(parameters[0])
    #     hpo_iteration_num = int(parameters[1])
    #     evaluation_indicator = parameters[2]
    #     classp = parameters[3]
    #     train_flag = bool(parameters[4])
    #     result_filename = None
    #     if 'as' in groups or 'AS' in groups:
    #         result_filename = groups[4]
    #     dataset = groups[7]
    #     alg_name, option, evaluation = ClassSelectionMain(dataset, alg_selection_time, hpo_iteration_num, evaluation_indicator, classp, train_flag)
    #     a = 1
