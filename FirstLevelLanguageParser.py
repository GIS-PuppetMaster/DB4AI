import re
from time import time

from Executor import Executor
from SecondLevelLanguageParser import Parser
from gdbc import GDBC

global_cursor = GDBC()
global_cursor.connect()

# 提前划分训练集和测试集以及xy
def get_sql_head(data_name):
    # sql = f"select TensorFromSql({data_name}_x) as data_x\n" \
    #       f"select TensorFromSql({data_name}_y) as data_y\n" \
    #       "select SHAPE(data_x) as sx\n" \
    #       "select 0.7 * sx[0] as train_len\n" \
    #       "select data_x[0:train_len,:] as x\n" \
    #       "select data_x[train_len:,:] as test_x\n" \
    #       "select data_y[0:train_len,:] as y\n" \
    #       "select data_y[train_len:,:] as test_y\n"
    sql = f"select TensorFromSql({data_name}_x) as x\n" \
          f"select TensorFromSql({data_name}_y) as y\n" \
          f"select TensorFromSql({data_name}_test_x) as test_x\n" \
          f"select TensorFromSql({data_name}_test_y) as test_y\n"
    return sql


def get_test_sql_head(data_name):
    sql = f"select TensorFromSql({data_name}_x) as test_x\n" \
          f"select TensorFromSql({data_name}_y) as test_y\n"
    return sql


def run_task_language(first_sql):
    # algorithm = 'logistic'
    # first_sql = f"select train(\'{algorithm}\') as model from real;"
    # first_sql = f"select test(\'{algorithm}\', \'model\') from real_multi_class_test;"
    # first_sql = "select * from real_x;"
    # parse
    filename_reg = '[a-zA-Z]+[a-zA-Z0-9_]*'
    train_match = f'^(select|SELECT)[ \t]+train([(].*[)])[ \t]+((as|AS)[ \t]+({filename_reg})[ \t]+)?(from|FROM)[ \t]*([ \t]*{filename_reg})[ \t]*;'
    test_match = f'^(select|SELECT)[ \t]+test([(].*[)])[ \t]+(from|FROM)[ \t]+([ \t]*{filename_reg}[ \t]*);'
    train_match = re.match(train_match, first_sql)
    test_match = re.match(test_match, first_sql)
    print(train_match)
    if train_match:
        groups = train_match.groups()
        algorithm = eval(groups[1])
        data_name = groups[6]
        if algorithm == 'logistic':
            sql = get_sql_head(data_name)
            sql += "create tensor lr(1,) from 0.1\n" \
                   "create tensor class_num(1,) from 2\n" \
                   "create tensor ridge(1,) from 0.005\n" \
                   "create tensor iter_times(1,) from 1000\n" \
                   "create tensor mse(1,)\n" \
                   "create tensor auc(1,)\n" \
                   "create tensor f1(1,)\n" \
                   "create tensor acc(1,)\n" \
                   "create tensor recall(1,)\n" \
                   "create tensor prec(1,)\n" \
                   "select logistic(acc,auc,prec,recall,mse,f1, test_x,test_y,x,y, ridge, lr, class_num, iter_times)"
        elif algorithm == 'Softmax':
            sql = get_sql_head(data_name)
            sql += "create tensor lr(1,) from 0.1\n" \
                   "create tensor class_num(1,) from 4\n" \
                   "create tensor ridge(1,) from 0.01\n" \
                   "create tensor iter_times(1,) from 1000\n" \
                   "create tensor mse(1,)\n" \
                   "create tensor auc(1,)\n" \
                   "create tensor f1(1,)\n" \
                   "create tensor acc(1,)\n" \
                   "create tensor recall(1,)\n" \
                   "create tensor prec(1,)\n" \
                   "select softmax_classification(acc,auc,prec,recall,mse,f1, test_x,test_y,x,y,class_num, ridge, lr, iter_times)"
        elif algorithm == "DNN":
            sql = get_sql_head(data_name)
            sql += "create tensor lr(1,) from 0.1\n" \
                   "create tensor class_num(1,) from 2\n" \
                   "create tensor iter_times(1,) from 1000\n" \
                   "create tensor mse(1,)\n" \
                   "create tensor auc(1,)\n" \
                   "create tensor f1(1,)\n" \
                   "create tensor acc(1,)\n" \
                   "create tensor recall(1,)\n" \
                   "create tensor prec(1,)\n" \
                   "create tensor layer_units(3,1) from zeros((3,1))\n" \
                   "select 0 as i\n" \
                   "loop(3){\n" \
                   "    select 4 as layer_units[i,...]\n" \
                   "    select i+1 as i\n" \
                   "}\n" \
                   "select DNN(acc,auc,prec,recall,mse,f1,test_x,test_y,x,y,lr,layer_units,iter_times,class_num)"
        else:
            raise Exception("unsupported algorithm")
        # path = f'operators/{algorithm}.sql'
        # with open(path, 'r', encoding='utf-8') as f:
        #     create_test = f.readlines()
        # Parser(create_test)()
        sql = sql.split("\n")
        for i in range(len(sql)):
            print(sql[i])
            sql[i] = sql[i] + "\n"
        result = Parser(sql)()
        # result.Show()
        executor = Executor(result)
        s = time()
        executor.run()
        return f"time cost:{time()-s} s"
    elif test_match:
        groups = test_match.groups()
        parameters = eval(groups[1])
        data_name = groups[3]
        algorithm = parameters[0]
        model = parameters[1]
        if algorithm == 'logistic':
            sql = get_test_sql_head(data_name)
            sql += "create tensor class_num(1,) from 2\n" \
                   "create tensor mse(1,)\n" \
                   "create tensor auc(1,)\n" \
                   "create tensor f1(1,)\n" \
                   "create tensor acc(1,)\n" \
                   "create tensor recall(1,)\n" \
                   "create tensor prec(1,)\n" \
                   "select test_logistic(acc,auc,prec,recall,mse,f1, test_x,test_y, class_num)"
        elif algorithm == 'Softmax':
            sql = get_test_sql_head(data_name)
            sql += "create tensor class_num(1,) from 4\n" \
                   "create tensor mse(1,)\n" \
                   "create tensor auc(1,)\n" \
                   "create tensor f1(1,)\n" \
                   "create tensor acc(1,)\n" \
                   "create tensor recall(1,)\n" \
                   "create tensor prec(1,)\n" \
                   "select test_softmax_classification(acc,auc,prec,recall,mse,f1, test_x,test_y,class_num)\n"
        elif algorithm == "DNN":
            sql = get_test_sql_head(data_name)
            sql += "create tensor class_num(1,) from 2\n"\
                   "create tensor mse(1,)\n" \
                   "create tensor auc(1,)\n" \
                   "create tensor f1(1,)\n" \
                   "create tensor acc(1,)\n" \
                   "create tensor recall(1,)\n" \
                   "create tensor prec(1,)\n" \
                   "create tensor layer_units(2,1) from zeros((2,1))\n" \
                   "select 0 as i\n" \
                   "loop(3){\n" \
                   "    select 4 as layer_units[i,...]\n" \
                   "    select i+1 as i\n" \
                   "}\n" \
                   "select test_DNN(acc,auc,prec,recall,mse,f1,test_x,test_y,layer_units,class_num)"
        else:
            raise Exception("unsupported algorithm")
        # path = f'operators/test_{algorithm}.sql'
        # with open(path, 'r', encoding='utf-8') as f:
        #     create_test = f.readlines()
        # Parser(create_test)()
        sql = sql.split("\n")
        for i in range(len(sql)):
            sql[i] = sql[i] + "\n"
        result = Parser(sql)()
        # result.Show()
        executor = Executor(result)
        s = time()
        executor.run()
        return f"time cost:{time() - s} s"
    else:
        try:
            global_cursor.execute(first_sql, True)
            print(global_cursor.fetch())
        except:
            raise Exception("sql error")
