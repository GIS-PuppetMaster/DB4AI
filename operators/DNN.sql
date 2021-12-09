operator DNN(acc,auc,prec,recall,mse,f1,test_x,test_y,x,y,lr,layer_units,iter_times){
    select SHAPE(x) as shape_x
    select shape_x[1] as feature_num
    select layer_units[0] as lu_0
    select layer_units[1] as lu_1
    select layer_units[2] as lu_2
    select layer_units[3] as lu_3
    create tensor w_0(feature_num,lu_0) from random((feature_num,lu_0),(0,1)) with grad
    create tensor b_0(1,lu_0) from zeros((1,lu_0)) with grad
    create tensor w_1(lu_0,lu_1) from random((lu_0,lu_1),(0,1)) with grad
    create tensor b_1(1,lu_1) from zeros((1,lu_1)) with grad
    create tensor w_2(lu_1,lu_2) from random((lu_1,lu_2),(0,1)) with grad
    create tensor b_2(1,lu_2) from zeros((1,lu_2)) with grad
    create tensor w_3(lu_2,lu_3) from random((lu_2,lu_3),(0,1)) with grad
    create tensor b_3(1,lu_3) from zeros((1,lu_3)) with grad
    create tensor w_4(lu_3,1) from random((lu_3,1),(0,1)) with grad
    create tensor b_4(1,1) from zeros((1,1)) with grad
    LOOP(iter_times){
        select 1/(1+EXP(-1 * (MATMUL(x,w_0)+b_0))) as output_0
        select 1/(1+EXP(-1 * (MATMUL(output_0,w_1)+b_1))) as output_1
        select 1/(1+EXP(-1 * (MATMUL(output_1,w_2)+b_2))) as output_2
        select 1/(1+EXP(-1 * (MATMUL(output_2,w_3)+b_3))) as output_3
        select 1/(1+EXP(-1 * (MATMUL(output_3,w_4)+b_4))) as output_4
        select MEAN(y*LOG(output_4)+(1-y)*LOG(1-output_4)) as loss
        select CleanGrad(w_0)
        select CleanGrad(b_0)
        select CleanGrad(w_1)
        select CleanGrad(b_1)
        select CleanGrad(w_2)
        select CleanGrad(b_2)
        select CleanGrad(w_3)
        select CleanGrad(b_3)
        select CleanGrad(w_4)
        select CleanGrad(b_4)
        select Backward(loss,w_0,b_0,w_1,b_1,w_2,b_2,w_3,b_3,w_4,b_4)
        select w_0-lr*GRADIENT(w_0) as w_0
        select b_0-lr*GRADIENT(b_0) as b_0
        select w_1-lr*GRADIENT(w_1) as w_1
        select b_1-lr*GRADIENT(b_1) as b_1
        select w_2-lr*GRADIENT(w_2) as w_2
        select b_2-lr*GRADIENT(b_2) as b_2
        select w_3-lr*GRADIENT(w_3) as w_3
        select b_3-lr*GRADIENT(b_3) as b_3
        select w_4-lr*GRADIENT(w_4) as w_4
        select b_4-lr*GRADIENT(b_4) as b_4
    }
    select 1/(1+EXP(-1 * (MATMUL(test_x,w_0)+b_0))) as output_0
    select 1/(1+EXP(-1 * (MATMUL(output_0,w_1)+b_1))) as output_1
    select 1/(1+EXP(-1 * (MATMUL(output_1,w_2)+b_2))) as output_2
    select 1/(1+EXP(-1 * (MATMUL(output_2,w_3)+b_3))) as output_3
    select 1/(1+EXP(-1 * (MATMUL(output_3,w_4)+b_4))) as pred
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
    select AUC(test_y, pred) as auc
    select ACC(test_y, pred) as acc
    select RECALL(test_y, pred) as recall
    select PRECISION(test_y, pred) as prec
    select MSE(test_y, pred) as mse
    select F1(test_y, pred) as f1

}