operator rbf_network(acc,auc,prec,recall,mse,f1, test_x, test_y, train_x, train_y,n_centers, n_classes,learning_rate, batch_size, iter_times){
    # reference: https://github.com/csnstat/rbfn/blob/master/RBFN.py
    select SHAPE(train_x) as x_shape
    select x_shape[0] as n_inputs
    select x_shape[1] as n_in
    create tensor centers(n_centers,n_in) from random((n_centers,n_in),(0,1)) with grad
    create tensor beta(1, n_centers) from ones((1,centers)) with grad
    create tensor w(n_centers, n_classes) from random((n_centers,n_classes),(0,1)) with grad
    create tensor b(1, n_classes) from random((1,n_classes),(0,1)) with grad
    loop(iter_times){
        create tensor i(1,) from 0
        loop(true){
            select centers as centers with grad
            select beta as beta with grad
            select i+batch_size as j
            if(j<=n_inputs){
                select train_x[i:j] as batch_x
                select train_y[i:j] as batch_y
            }
            else{
                select train_x[i:] as batch_x
                select train_y[i:] as batch_y
            }
            select SHAPE(batch_x) as x_shape
            select x_shape[0] as n_input
            # (n_input, n_centers, n_in)
            select REPEAT(centers, n_input, 1, 1) as A with grad
            # (n_input, n_centers, n_in)
            select REPEAT(UNSQUEEZE(batch_x, 1), 1, n_centers, 1) as B with grad
            # (n_input, n_centers)
            select EXP(0-beta*SQRT(SUM(POW(A-B,2),2))) as C with grad
            select MATMUL(C,w)+b as class_score with grad
            select 0-MEAN(batch_y*LOG(class_score)) as loss with grad
            select CleanGrad(centers, beta, w, b)
            select Backward(loss, centers, beta, w, b)
            select GRADIENT(centers) as g_centers
            select GRADIENT(beta) as g_beta
            select GRADIENT(w) as g_w
            select GRADIENT(b) as g_b
            update centers-learning_rate*g_centers as centers
            update beta-learning_rate*g_beta as beta
            update w-learning_rate*g_w as w
            update b-learning_rate*g_b as b
            select j as i
            if(i>=n_inputs){
                break
            }
        }
    }
    select SHAPE(test_x) as x_shape
    select x_shape[0] as n_input
    select REPEAT(centers, n_input, 1, 1) as A
    select REPEAT(UNSQUEEZE(test_x, 1), 1, n_centers, 1) as B
    select EXP(0-beta*SQRT(SUM(POW(A-B,2),2))) as C
    select MATMUL(C,w)+b as pred
    select AUC(test_y, pred) as auc
    select ACC(test_y, pred) as acc
    select RECALL(test_y, pred) as recall
    select PRECISION(test_y, pred) as prec
    select MSE(test_y, pred) as mse
    select F1(test_y, pred) as f1
}