create tensor x(1000,4) from random((1000,4),(0,1))
create tensor y(1000,1) from random((1000,1),(0,1),'int')
create tensor w(4,1) from random((4,1),(0,1)) with grad
create tensor learning_rate(1,) from 0.01
create tensor threshold(1,) from 0
create tensor iter_times(1,) from 10
select logistic(x,y,w, learning_rate, threshold, iter_times)


