select TensorFromSql(real_x) as x
select TensorFromSql(real_y) as y
select TensorFromSql(real_test_x) as test_x
select TensorFromSql(real_test_y) as test_y
create tensor lr(1,) from 0.01
create tensor class_num(1,) from 2
create tensor ridge(1,) from 0.005
create tensor iter_times(1,) from 500
create tensor mse(1,)
create tensor auc(1,)
create tensor f1(1,)
create tensor acc(1,)
create tensor recall(1,)
create tensor prec(1,)
select logistic(acc,auc,prec,recall,mse,f1, test_x,test_y,x,y, ridge, lr, class_num, iter_times)
# select SaveTable(auc, logistic_auc, print)
# select SaveTable(acc, logistic_acc, print)
# select SaveTable(recall, logistic_recall, print)
# select SaveTable(prec, logistic_prec, print)
# select SaveTable(mse, logistic_mse, print)
# select SaveTable(f1, logistic_f1, print)
# 测试通过

