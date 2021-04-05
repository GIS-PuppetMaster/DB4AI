create tensor x(1000,4) from random((1000,4),(0,1))
create tensor y(1000,1) from zeros((1000,1))
select Argmax(x, 1) as tmp
create tensor i(1,) from 0
loop(1000){
    select tmp[i] as t
    if(t>1){
        select 1 as y[i,...]
    }
    select i+1 as i
}
create tensor w(4,1) from random((4,1),(0,1)) with grad
create tensor lr(1,) from 0.01
create tensor threshold(1,) from 0.3
create tensor iter_times(1,) from 10000
select logistic(x,y,w, lr, threshold, iter_times)


