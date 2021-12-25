select TensorFromSql(real_x) as x
select TensorFromSql(real_y) as y
select TensorFromSql(real_test_x) as test_x
select TensorFromSql(real_test_y) as test_y
create tensor lr(1,) from 0.01
create tensor class_num(1,) from 2
create tensor iter_times(1,) from 500
create tensor mse(1,)
create tensor auc(1,)
create tensor f1(1,)
create tensor acc(1,)
create tensor recall(1,)
create tensor prec(1,)
create tensor layer_units(3,1) from zeros((3,1))
select 0 as i
loop(3){
    select 4 as layer_units[i,...]
    select i+1 as i
}
select DNN(acc,auc,prec,recall,mse,f1,test_x,test_y,x,y,lr,layer_units,iter_times,class_num)