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
create tensor k(1,) from 10
select KNN(test_x, x, y, k)
