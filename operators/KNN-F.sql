operator KNNF(acc,auc,prec,recall,mse,f1, data_input, test_y, x, y, k){
    select MAX(y)+1 as class_number
    select SHAPE(data_input) as data_input_shape
    select data_input_shape[0] as data_input_len
    create tensor data_input_index(1,) from 0
    create tensor answer(data_input_len,) from full((data_input_len,),-1)
    select SHAPE(x) as x_shape
    select x_shape[0] as data_len
    select x_shape[1] as data_width
    LOOP(data_input_len){
        select data_input[data_input_index] as data
        select POW(x-data,2) as dis
        create tensor distance(data_len,) from full((data_len, ), 0.0)
        create tensor i(1,) from 0
        LOOP(data_len){
            create tensor j(1,) from 0
            create tensor s(1,) from 0
            LOOP(data_width){
                select s+dis[i,j] as s
                select j+1 as j
            }
            select SQRT(s) as distance[i]
            select 1-distance[i] as weight[i]
            select i+1 as i
        }
        select ARGSORT(distance) as index_order
        create tensor k_iter(1,) from 0
        create tensor classes(class_number,) from full((class_number,), 0)
        LOOP(k){
            select index_order[k_iter] as ink
            select y[ink] as yink
            select classes[yink]+weight[yink] as classes[yink]
            select k_iter+1 as k_iter
        }
        select REVERSE(ARGSORT(classes)) as classes_order
        select classes_order[0] as answer[data_input_index]
        select data_input_index+1 as data_input_index
    }
    select AUC(test_y, answer) as auc
    select ACC(test_y, answer) as acc
    select RECALL(test_y, answer) as recall
    select PRECISION(test_y, answer) as prec
    select MSE(test_y, answer) as mse
    select F1(test_y, answer) as f1
}