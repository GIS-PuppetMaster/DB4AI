select TensorFromSql(real_x) as x
select TensorFromSql(real_y) as tmp_y
select TensorFromSql(real_test_x) as test_x
select TensorFromSql(real_test_y) as tmp_test_y
create tensor y(500,2) from zeros((500,2))
create tensor i(1,) from 0
loop(500){
    if(tmp_y[i]==1){
        select 1 as y[i,0]
    }
    else{
        select 1 as y[i,1]
    }
    select i+1 as i
}
create tensor test_y(69,2) from zeros((69,2))
select 0 as i
loop(69){
    if(tmp_test_y[i]==1){
        select 1 as test_y[i,0]
    }
    else{
        select 1 as test_y[i,1]
    }
    select i+1 as i
}
create tensor lr(1,) from 0.01
create tensor class_num(1,) from 2
create tensor iter_times(1,) from 2000
create tensor mse(1,)
create tensor auc(1,)
create tensor f1(1,)
create tensor acc(1,)
create tensor recall(1,)
create tensor prec(1,)
create tensor layer_units(3,1) from zeros((3,1))
select 0 as i
loop(3){
    select 4-i as layer_units[i,...]
    select i+1 as i
}
select DNN(acc,auc,prec,recall,mse,f1,test_x,test_y,x,y,lr,layer_units,iter_times,class_num)