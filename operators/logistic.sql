operator logistic(acc,auc,prec,recall,mse,f1, test_x,test_y,x,y, ridge, learning_rate, iter_times){
    # 二分类, https://github.com/csnstat/rbfn/blob/master/RBFN.py
    select SHAPE(x) as sx
    select sx[0] as record_num
    select sx[1] as feature_num
    create tensor w(feature_num,1) from RANDOM((feature_num,1),(0,1)) with grad
    create tensor hx(feature_num, class_num)
    LOOP(iter_times){
        SELECT w as w with grad
        select hx as hx with grad
        select 1/(1+POW(CONSTANT.E, MATMUL(x, w))) as hx with grad
        SELECT ridge*MEAN(POW(w,2))-MEAN(y * LOG(hx) + (1 - y) * LOG(1 - hx)) AS loss with grad
        select Backward(loss)
        SELECT GRADIENT(w) AS g
        SELECT w-learning_rate * g AS w
    }
    select 1/(1+POW(CONSTANT.E, MATMUL(x, w))) as pred
    create tensor i(1,) from 0
    LOOP(record_num){
        if(hx[i,:]>=0.5){
            select 1 as hx[i,:]
        }
        else{
            select 0 as hx[i,:]
        }
        select i+1 as i
    }
    select AUC(test_y, pred) as auc
    select ACC(test_y, pred) as acc
    select RECALL(test_y, pred) as recall
    select PRECISION(test_y, pred) as prec
    select MSE(test_y, pred) as mse
    select F1(test_y, pred) as f1
}
