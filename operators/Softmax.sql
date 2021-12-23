operator softmax_classification(acc,auc,prec,recall,mse,f1,test_x,test_y,x,y,class_num, ridge, learning_rate, iter_times){
    # 多分类, 要求y的shape为(record_num, class_num)
    select SHAPE(x) as sx
    select sx[0] as record_num
    select sx[1] as feature_num
    create tensor w(feature_num,class_num) from RANDOM((feature_num,class_num),(0,1)) with grad
    create tensor b(1,class_num) from RANDOM((1,class_num),(0,1)) with grad
    create tensor pred(feature_num, class_num) with pred
    LOOP(iter_times){
        select Softmax(MATMUL(x,w)+b, 1) as pred with grad
        select -MEAN(y*LOG(pred)) as loss with grad
        select CleanGrad(w)
        select Backward(loss, w)
        SELECT GRADIENT(w) AS g
        update w-learning_rate * g AS w
    }
    select SaveTable(w, softmax_w, null)
    select SaveTable(b, softmax_b, null)
    select Softmax(MATMUL(test_x,w)+b, 1) as pred
    if(class_num>2){
        select Argmax(test_y,1) as test_y
        select Argmax(pred,1) as pred
    }
    # select AUC(test_y, pred) as auc
    select ACC(test_y, pred) as acc
    # select RECALL(test_y, pred) as recall
    # select PRECISION(test_y, pred) as prec
    # select MSE(test_y, pred) as mse
    # select F1(test_y, pred) as f1
    select SaveTable(acc, softmax_acc, print)
    create tensor a(1,1)
}
# 测试通过