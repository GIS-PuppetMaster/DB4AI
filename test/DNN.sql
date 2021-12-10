create tensor x(100,4) from random((100,4),(0,1))
create tensor y(100,1) from zeros((100,1))
select Argmax(x, 1) as tmp
create tensor i(1,) from 0
create tensor t(1,) from 0
loop(100){
    select tmp[i] as t
    if(t>1){
         select 1 as y[i,...]
    }
    select i+1 as i
}
create tensor test_x(10,4) from random((10,4),(0,1))
create tensor test_y(10,1) from zeros((10,1))
select Argmax(test_x, 1) as tmp
select 0 as i
loop(10){
    select tmp[i] as t
    if(t>1){
         select 1 as test_y[i,...]
    }
    select i+1 as i
}
create tensor lr(1,) from 0.005
create tensor class_num(1,) from 2
create tensor iter_times(1,) from 1000
create tensor mse(1,)
create tensor auc(1,)
create tensor f1(1,)
create tensor acc(1,)
create tensor recall(1,)
create tensor prec(1,)
create tensor layer_units(4,1) from zeros((4,1))

select 0 as i
loop(4){
    select 5-i as layer_units[i,...]
    select i+1 as i
}
select DNN(acc,auc,prec,recall,mse,f1,test_x,test_y,x,y,lr,layer_units,iter_times)