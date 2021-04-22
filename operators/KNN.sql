operator KNN(data_input, x, y, k){
    select MAX(y)+1 as class_number
    select SHAPE(data_input) as data_input_shape
    select data_input_shape[0] as data_input_len
    create tensor data_input_index(1,) from 0
    create tensor answer(data_input_len,) from -1
    select SHAPE(x) as x_shape
    select x_shape[0] as data_len
    select x_shape[1] as data_width
    LOOP(data_input_len){
        select data_input[data_input_index] as data
        select POW(x-data,2) as dis
        create tensor distance(data_width,) from 0
        create tensor i(1,) from 0
        LOOP(data_len){
            create tensor j(1,) from 0
            create tensor s(1,) from 0
            LOOP(data_width){
                select s+dis[i][j] as s
                select j+1 as j
            }
            select SQRT(s) as distance[i]
            select i+1 as i
        }
        select REVERSE(ARGSORT(distance)) as index_order
        create tensor k_iter(1,) from 0
        create tensor classes(class_number,) from 0
        LOOP(k){
            select index_order[k] as
            select classes[y[index_order[k]]]+1 as classes[y[index_order[k]]]
            select k_iter+1 as k_iter
        }
        select REVERSE(SORT(classes)) as classes_order
        select classes_order[0] as answer[data_input_index]
        select data_input_index+1 as data_input_index
    }
    select answer
}