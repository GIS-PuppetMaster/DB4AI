create tensor x(10,4) from random((10,4),(0,1))
create tensor y(10,1) from zeros((10,1))
select Argmax(x, 1) as tmp
create tensor i(1,) from 0
create tensor temp(1,) from 0
loop(10){
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
create tensor lr(1,) from 0.01
create tensor class_num(1,) from 2
create tensor ridge(1,) from 0.01
create tensor iter_times(1,) from 100
create tensor mse(1,)
create tensor auc(1,)
create tensor f1(1,)
create tensor acc(1,)
create tensor recall(1,)
create tensor prec(1,)
select logistic(acc,auc,prec,recall,mse,f1, test_x,test_y,x,y, ridge, lr, class_num, iter_times)
# 测试通过

