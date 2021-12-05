operator test_softmax_classification(acc,auc,prec,recall,mse,f1,test_x,test_y,class_num){
    # 多分类, 要求y的shape为(record_num, class_num)
    select TensorFromSql(softmax_w) as w
    select TensorFromSql(softmax_b) as b
    select Softmax(MATMUL(test_x,w)+b, 1) as pred
    if(class_num>2){
        select Argmax(test_y,1) as test_y
        select Argmax(pred,1) as pred
    }
    select AUC(test_y, pred) as auc
    select ACC(test_y, pred) as acc
    select RECALL(test_y, pred) as recall
    select PRECISION(test_y, pred) as prec
    select MSE(test_y, pred) as mse
    select F1(test_y, pred) as f1
}
# 测试通过