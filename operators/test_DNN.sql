operator test_DNN(acc,auc,prec,recall,mse,f1,test_x,test_y,layer_units,class_num){
    select layer_units[0] as lu_0
    select layer_units[1] as lu_1
    select TensorFromSql(w_0_dnn) as w_0
    select TensorFromSql(b_0_dnn) as b_0
    select TensorFromSql(w_1_dnn) as w_1
    select TensorFromSql(b_1_dnn) as b_1
    select TensorFromSql(w_2_dnn) as w_2
    select TensorFromSql(b_2_dnn) as b_2
    select LeakyRelu(MATMUL(test_x,w_0)+b_0) as output_0
    select LeakyRelu(MATMUL(output_0,w_1)+b_1) as output_1
    select 1/(1+EXP(-1 * (MATMUL(output_1,w_2)+b_2))) as pred
    create tensor i(1,) from 0
    select SHAPE(test_x) as sx
    select sx[0] as record_num
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
    select SaveTable(acc, test_acc_dnn, print)
    create tensor a(1,1)
}