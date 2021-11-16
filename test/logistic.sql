select SQL('select feature_1,feature_2,feature_3 from train_data') as train_x
select SQL('select label from train_data') as train_y
select SQL('select feature_1,feature_2,feature_3 from test_data') as test_x
select SQL('select label from test_data') as test_y
create tensor lr(1,) from 0.01
create tensor class_num(1,) from 2
create tensor ridge(1,) from 0.0001
create tensor iter_times(1,) from 50
create tensor mse(1,)
create tensor auc(1,)
create tensor f1(1,)
create tensor acc(1,)
create tensor recall(1,)
create tensor prec(1,)
select logistic(acc,auc,prec,recall,mse,f1, test_x,test_y,x,y, ridge, lr, class_num, iter_times)
# 测试通过

