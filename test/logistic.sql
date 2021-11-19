select TensorFromSql('train_data') as train
select TensorFromSql('test_data') as test
select train[:,:-1] as train_x
select train[:,-1] as train_y
select test[:,:-1] as test_x
select test[:,-1] as test_y
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

