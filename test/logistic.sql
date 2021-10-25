create tensor x(1000,4) from random((1000,4),(0,1))
create tensor y(1000,1) from zeros((1000,1))
select Argmax(x, 1) as tmp
create tensor i(1,) from 0
create tensor temp(1,) from 0
loop(1000){
    select tmp[i] as t
    if(t>1){
        select 1 as y[i,...]
    }
    select i+1 as i
}
create tensor test_x(100,4) from random((100,4),(0,1))
create tensor test_y(100,1) from zeros((100,1))
select Argmax(test_x, 1) as tmp
select 0 as i
loop(100){
    select tmp[i] as t
    if(t>1){
        select 1 as test_y[i,...]
    }
    select i+1 as i
}
create tensor lr(1,) from 0.01
create tensor class_num(1,) from 2
create tensor ridge(1,) from 0.01
create tensor iter_times(1,) from 10000
create tensor mse(1,)
create tensor auc(1,)
create tensor f1(1,)
create tensor acc(1,)
create tensor recall(1,)
create tensor prec(1,)
select logistic(acc,auc,prec,recall,mse,f1, test_x,test_y,x,y, ridge, lr, class_num, iter_times)
# 测试通过

