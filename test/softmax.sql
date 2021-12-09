create tensor x(1000,4) from random((1000,4),(0,1))
create tensor y(1000,3) from zeros((1000,3))
select Argmax(x, 1) as tmp
create tensor i(1,) from 0
create tensor temp(1,) from 0
loop(1000){
    select tmp[i] as t
    if(t==2){
        select 1 as y[i,1]
    }
    elif(t>2){
        select 1 as y[i,2]
    }
    else{
        select 1 as y[i,0]
    }
    select i+1 as i
}
create tensor test_x(100,4) from random((100,4),(0,1))
create tensor test_y(100,3) from zeros((100,3))
select Argmax(test_x, 1) as tmp
select 0 as i
loop(100){
    select tmp[i] as t
    if(t==2){
        select 1 as test_y[i,1]
    }
    elif(t>2){
        select 1 as test_y[i,2]
    }
    else{
        select 1 as test_y[i,0]
    }
    select i+1 as i
}
create tensor lr(1,) from 0.01
create tensor class_num(1,) from 3
create tensor ridge(1,) from 0.01
create tensor iter_times(1,) from 500
create tensor mse(1,)
create tensor auc(1,)
create tensor f1(1,)
create tensor acc(1,)
create tensor recall(1,)
create tensor prec(1,)
select softmax_classification(acc,auc,prec,recall,mse,f1, test_x,test_y,x,y,class_num, ridge, lr, iter_times)
# select SaveTable(auc, Softmax_auc, print)
# select SaveTable(acc, Softmax_acc, print)
# select SaveTable(recall, Softmax_recall, print)
# select SaveTable(prec, Softmax_prec, print)
# select SaveTable(mse, Softmax_mse, print)
# select SaveTable(f1, Softmax_f1, print)


