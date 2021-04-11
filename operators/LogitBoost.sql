operator logit_boost(x,y,J,M){
    # y的取值范围为0, 1
    select SHAPE(x)[0] as N
    create tensor F(J, ) from zeros((J, ))
    create tensor p(N,J) from full((N,J), 1/J)
    create tensor w(N,J) from full((N,J), 1/N)
    create tensor f(M,J) from zeros((M,J))

    create tensor m(1,) from 0
    loop(M){
        create tensor j(1,) from 0
        loop(J){
            select p[:,j] as p_j
            select 1-p_j as tmp
            select p_j * tmp as tmp2
            select tmp2 as w[:,j]
            select (y[:,j]-p_j)/tmp2 as z_j
            # 新增加权最小二乘算子, 输出为最小二乘的a和b, y=a+bx
            select WLS(x, z, w) as f[m,j]
            select j+1 as j
        }
        select (J-1)/J*(f[m,j]-1/J*SUM(f[m,])) as f[m,j]
        select F[j] + f[m,j] as F[j]
        select EXP(F[j])/SUM(EXP(F)) as p[j]
        select m+1 as m
    }
}