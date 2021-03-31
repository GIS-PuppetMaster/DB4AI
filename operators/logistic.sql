operator logistic(x,y,w,learning_rate,threshold, iter_times){
    LOOP(iter_times){
        SELECT 1 / (1 + POW(CONSTANT.E, MATMUL(x, w))) AS hx FROM w, x with grad
        SELECT y * LOG(hx) + (1 - y) * (1 - hx) AS loss FROM y, hx with grad
        SELECT GRADIENT(loss, w) AS g FROM loss, w
        SELECT learning_rate * g + w AS w FROM learning_rate, g, w
        IF(loss<threshold){
            BREAK
        }
    }
}
