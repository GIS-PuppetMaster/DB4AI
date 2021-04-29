operator softmax_classification(acc,auc,prec,recall,mse,f1,test_x,test_y,x,y,hidden, ridge, learning_rate, iter_times){
    # 多分类
    select SHAPE(x) as sx
    select sx[0] as record_num
    select sx[1] as feature_num
    create tensor w(feature_num,hidden) from RANDOM((feature_num,hidden),(0,1)) with grad
    create tensor b(1,hidden) from RANDOM((1,hidden),(0,1)) with grad
    create tensor pred(feature_num, hidden)
    LOOP(iter_times){
        SELECT w as w with grad
        SELECT b as b with grad
        select pred as pred with grad
        select Softmax(MATMUL(x,w)+b) as pred with grad
        select 0-MEAN(y*LOG(pred)) as loss with grad
        select Backward(loss)
        SELECT GRADIENT(w) AS g
        SELECT w-learning_rate * g AS w
    }
    select Argmax(Softmax(MATMUL(x,w)+b),1) as pred
    select AUC(test_y, pred) as auc
    select ACC(test_y, pred) as acc
    select RECALL(test_y, pred) as recall
    select PRECISION(test_y, pred) as prec
    select MSE(test_y, pred) as mse
    select F1(test_y, pred) as f1
}
