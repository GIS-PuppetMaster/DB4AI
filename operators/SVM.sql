operator linear_kernel(linear_kernel_value, x1, x2){
    select TRANSPOSE(x1) * x2 as linear_kernel_value
}

operator SVM_fast_predict(y_pred, non_zero_a, x_a, y_a, b, x){
    select SUM(non_zero_a * y_a * (x_a * x)) + b as y_pred
}

operator SVM_predict(y_pred,w,b,x){
    select TRANSPOSE(w)*x+b as y_pred
}

operator take_step(i, j, w,b, a, x, y, c, eps, kernel_cache, error_cache){
    if(i!=j){
        select a[i] as alpha1
        select a[j] as alpha2
        select x[i] as x1
        select x[j] as x2
        select y[i] as y1
        select y[j] as y2
        select error_cache[i] as e1
        select error_cache[j] as e2
        select y1 * y2 as s
        select MAX(0, alpha2-alpha1) as l
        select MIN(c, c+alpha2-alpha1) as h
        if(l!=h){
            select kernel_cache[i,i] as k11
            select kernel_cache[i,j] as k12
            select kernel_cache[j,j] as k22
            select k11+k22 - 2 * k12 as eta
            if(eta>0){
                select alpha2 + y2*(e1-e2)/eta as a2
                if(a2<l){
                    select a2 as l
                }
                elif(a2>h){
                    select a2 as h
                }
            }
            else{
                select y1*(e1+b)-alpha1*k11-s*alpha2*k12 as f1
                select y2*(e2+b)-s*alpha1*k12-alpha2*k22 as f2
                select alpha1+s*(alpha2-l) as l1
                select alpha1+s*(alpha2-h) as h1
                select l1*f1+l*f2+0.5*POW(l1,2)*k11+0.5*POW(l,2)*k22+s*l*l1*k12 as Lobj
                select h1*f1+h*f2+0.5*POW(h1,2)*k11+0.5*POW(h,2)*k22+s*h*h1*k12 as Hobj
                if(Lobj < Hobj-eps){
                    select l as a2
                }
                elif(Lobj > Hobj+eps){
                    select h as a2
                }
                else{
                    select alpha2 as a2
                }
            }
            select Abs(a2-alpha2) as l_tmp
            select eps*(a2+alpha2+eps) as r_tmp
            if(l_tmp>=r_tmp){
                select a2-alpha2 as delta_a2
                select alpha1+s*(0-delta_a2) as a1
                select a1-alpha1 as delta_a1
                select e1+y1*delta_a1*k11+y2*delta_a2*k12+b as b[i]
                select e2+y1*delta_a1*k12+y2*delta_a2*k22+b as b[j]
                select w+y1*delta_a1*x1 + y2*delta_a2*x2 as w
                select SVM_predict(w,b,x1) as y_pred_1
                select SVM_predict(w,b,x2) as y_pred_2
                select y_pred_1 - y1 as error_cache[i]
                select y_pred_2 - y2 as error_cache[j]
                select a1 as a[i]
                select a2 as a[j]
            }
        }
    }
}


operator SVM(x, y, c, eps, iter_times){
    select SHAPE(x)[0] as n
    select SHAPE(x)[1] as m
    create tensor kernel_cache(n,n)
    create tensor error_cache(n,)
    create tensor w(m,) from RANDOM((m,),(0,1))
    create tensor b(n,) from RANDOM((n,),(0,1))
    create tensor a(n,) from 0
    create tensor i from 0
    LOOP(n){
        create tensor j from 0
        LOOP(n){
            select linear_kernel(x[i,:],x[j:]) as kernel_cache[i,j]
            select j+1 as j
        }
        select i+1 as i
    }
    select 0 as t
    create tensor g(n,)
    select SUM(a*y*x) as tmp
    LOOP(iter_times){
        create tensor j(1,) from 0
        loop(n){
            select tmp*x[j]+b as g[j]
            select j+1 as j
        }
        select y*g as tmp2
        select POW(tmp2-1,2) as c1
        select Deepcopy(c1) as c2
        select Deepcopy(c1) as c3
        select 0 as j
        loop(n){
            if(a[j]>0 or tmp2[j]>=1){
                select 0 as c1[j]
            }
            elif(a[j]==0 or a[j]==c or tmp2[j]==1){
                select 0 as c2[j]
            }
            elif(a[j]<C or tmp2[j]<=1){
                select 0 as c3[j]
            }
            select j+1 as j
        }
        select ARGMAX(c1+c2+c3) as i
        select random((1,),(0,n),'uniform') as j
        select take_step(i,j,w,b,a,x,y,eps,kernel_cache,error_cache)
        select t+1 as t
    }
}