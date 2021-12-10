create tensor x(1000,4) from random((1000,4),(0,1))
create tensor y(1000,2) from zeros((1000,2))
select Argmax(x, 1) as tmp
create tensor i(1,) from 0
create tensor t(1,) from 0
loop(1000){
    select tmp[i] as t
    if(t>1){
        select 1 as y[i,0]
    }
    else{
        select 1 as y[i,1]
    }
    select i+1 as i
}
create tensor test_x(100,4) from random((100,4),(0,1))
create tensor test_y(100,2) from zeros((100,2))
select Argmax(test_x, 1) as tmp
select 0 as i
loop(100){
    select tmp[i] as t
    if(t>1){
        select 1 as test_y[i,0]
    }
    else{
        select 1 as test_y[i,1]
    }
    select i+1 as i
}
create tensor lr(1,) from 0.01
create tensor class_num(1,) from 2
create tensor iter_times(1,) from 100
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