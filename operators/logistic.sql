operator logistic(x,y,class_num, ridge, learning_rate, iter_times){
    select SHAPE(x) as sx
    select sx[0] as record_num
    select sx[1] as feature_num
    select class_num-1 as tmp
    create tensor w(feature_num,tmp) from RANDOM((feature_num,tmp),(0,1)) with grad
    create tensor px(feature_num, class_num)
    LOOP(iter_times){
        SELECT w as w with grad
        select POW(CONSTANT.E, MATMUL(x, w[:,:-1])) as tmp1
        create tensor px(record_num, tmp) with grad
        SELECT tmp1 / SUM((tmp1+1), 1) AS px[:,:-1] with grad
        select 1-(SUM(px[:,:-1], 1)) as px[:,-1] with grad
        SELECT ridge*POW(w,2)-SUM(y * LOG(px) + (1 - y) * LOG(1 - px)) AS loss with grad
        SELECT GRADIENT(loss, w) AS g
        SELECT w+learning_rate * g AS w
    }
}
