create tensor x(1000,4) from random((1000,4),(0,1))
create tensor y(1000,1) from zeros((1000,1))
create tensor c(1, ) from 1
create tensor eps(1, ) from 0.5
create tensor iter_times(1, ) from 10
select Argmax(x, 1) as tmp
create tensor i(1,) from 0
loop(1000){
    select tmp[i] as t
    if(t>1){
        select 1 as y[i,...]
    }
    select i+1 as i
}
select SVM(x, y, c, eps, iter_times)