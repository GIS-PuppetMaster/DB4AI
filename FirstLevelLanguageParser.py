import re
from AutoCash_API.ClassSelectionMain import *

if __name__ == '__main__':

    # select AutoClassify(alg_selection_time:float, hpo_iteration_num:int, evaluation_indicator:[], classp, train_flag:bool) [as result_filename] from filename
    # select train(result_filename, classp, evaluation_indicator) from dataset
    # first_sql = input('please input ML training task\n')
    first_sql = 'select AutoClassify(60, 10, \'acc\', \'end\', \'false\') from filename'
    # parse
    filename_reg = '[a-zA-Z]+[a-zA-Z0-9_]*'
    auto_classify_reg = f'^(select|SELECT)[ \t]+AutoClassify([(].*[)])[ \t]+((as|AS)[ \t]+({filename_reg})[ \t]+)?((from|FROM)[ \t]+({filename_reg}))'
    train_match = f'^(select|SELECT)[ \t]+train[(].*[)][ \t]+((from|FROM)[ \t]+{filename_reg})'
    auto_match = re.match(auto_classify_reg, first_sql)
    train_match = re.match(train_match, first_sql)
    if auto_match:
        groups = auto_match.groups()
        parameters = eval(groups[1])
        alg_selection_time = float(parameters[0])
        hpo_iteration_num = int(parameters[1])
        evaluation_indicator = parameters[2]
        classp = parameters[3]
        train_flag = bool(parameters[4])
        result_filename = None
        if 'as' in groups or 'AS' in groups:
            result_filename = groups[4]
        dataset = groups[7]
        alg_name, option, evaluation = ClassSelectionMain(dataset, alg_selection_time, hpo_iteration_num, evaluation_indicator, classp, train_flag)
        a = 1
