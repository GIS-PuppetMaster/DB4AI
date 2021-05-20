create tensor x(1000,3) from random((1000,3),(0,1))
create tensor y(1000,1) from zeros((1000,1))
select Argmax(x, 1) as tmp
create tensor i(1,) from 0
loop(1000){
    select tmp[i] as t
    if(t>1 and t<3){
        select 1 as y[i,...]
    }
    select i+1 as i
}
create tensor test_x(10,3) from random((10,3),(0,1))
create tensor test_y(10,1) from zeros((10,1))
select Argmax(test_x, 1) as tmp
select 0 as i
loop(10){
    select tmp[i] as t
    if(t>1 and t<3){
        select 1 as test_y[i,...]
    }
    select i+1 as i
}
create tensor k(1,) from 10
create tensor mse(1,)
create tensor auc(1,)
create tensor f1(1,)
create tensor acc(1,)
create tensor recall(1,)
create tensor prec(1,)
select KNN(acc,auc,prec,recall,mse,f1, test_x, test_y, x, y, k)
# 测试通过