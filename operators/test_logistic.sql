operator test_logistic(acc,auc,prec,recall,mse,f1, test_x,test_y, class_num){
    # 二分类, https://github.com/csnstat/rbfn/blob/master/RBFN.py
    select SHAPE(test_x) as sx
    select sx[0] as record_num
    select sx[1] as feature_num
    select TensorFromSql(w_logistic) as w
    select TensorFromSql(hx_logistic) as hx
    select 1/(1+POW(CONSTANT.E, -1 * MATMUL(test_x, w))) as pred
    create tensor i(1,) from 0
    LOOP(record_num){
        if(pred[i]>=0.5){
            select 1 as pred[i]
        }
        else{
            select 0 as pred[i]
        }
        select i+1 as i
    }
    # select AUC(test_y, pred) as auc
    select ACC(test_y, pred) as acc
    # select RECALL(test_y, pred) as recall
    # select PRECISION(test_y, pred) as prec
    # select MSE(test_y, pred) as mse
    # select F1(test_y, pred) as f1
    # select SaveTable(auc, test_auc_logisitc, print)
    select SaveTable(acc, test_acc_logisitc, print)
    # select SaveTable(recall, test_recall_logisitc, print)
    # select SaveTable(prec, test_prec_logisitc, print)
    # select SaveTable(mse, test_mse_logisitc, print)
    # select SaveTable(f1, test_f1_logisitc, print)
    create tensor a(1,1)
}
# 测试通过