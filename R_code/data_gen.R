if(model_type == "reg_P111"){
    p = 1
    set.seed(1001)
    gen_true = function(X,Z){
        n1 = length(X)
        s = base::sample(c(-1,1,0),n1,replace=TRUE) * !(X >=-5 & X<= 5)
        #y = sin(X[,1]/2) + Z*(sigma0 + sqrt(abs(X[,1]/5)))
        y = (X*s + Z) #* !(X >=-5 & X<= 5)
        y = matrix(y,length(y),1)
        return(y)
    }
    
    z_true = function(x,n){zz = rnorm(n, mean=0, sd=sigma0*sqrt(abs(x[,1]*0.25)));return(zz)}
    
    X = matrix(runif(n*p, -20, 20), ncol=p)
    Z = z_true(X,n)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<5000){n_test=0.1*n}
    if(n>=5000&n<=10000){n_test=500}
    Xt = matrix(seq(-20, 20, length=n_test), ncol=p, nrow=n_test)
    Z = z_true(Xt,n_test)
    yt = gen_true(Xt,Z)
}
if(model_type == "reg_largeP"){
    
    bt0 = seq(-3,3,length.out=p)
    
    gen_true = function(X,Z){
        bt0 = seq(-3,3,length.out=p)
        y = X%*%bt0  + Z
        y = matrix(y,length(y),1)
        return(y)
    }
    
    z_true = function(x,n){zz = rnorm(n, mean=0, sd=sigma0);return(zz)}
    
    set.seed(2022)
    X = matrix(rnorm(n*p), ncol=p)
    Z = rnorm(n, mean=0, sd=sigma0)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<=5000){n_test=0.1*n}
    if(n>5000&n<=10000){n_test=500}
    Xt = matrix(rnorm(n_test*p), ncol=p)
    Xt[1,] = c(1, rep(0,p-1))
    Xt[2,] = c(0, 1, rep(0,p-2))
    Xt[3,] = c(0, 0, 1, rep(0, p-3))
    Z = rnorm(n_test, mean=0, sd=sigma0)
    yt = gen_true(Xt,Z)
}
if(model_type == "reg_nonparam"){
    
    bt0 = seq(-2,2,length.out=p)
    
    gen_true = function(X,Z){
        bt0 = seq(-2,2,length.out=p)
        X = matrix(X)
        y = 0.5*log(abs(10-(X[1,])^2))+0.75*exp(X[2,]*X[3,]/5)-0.25*abs(X[4,]/2)+Z
        y = matrix(y,length(y),1)
        return(y)
    }
    
    z_true = function(x,n){zz = rnorm(n, mean=0, sd=sigma0);return(zz)}
    
    #Seed = 128783
    set.seed(Seed)
    X = matrix(rnorm(n*p), ncol=p)
    Z = z_true(X,n)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<=5000){
        n_test=0.1*n
        #if(fac > 1){n_test=0.2*n}
    }
    if(n>5000&n<=10000){n_test=500}
    Xt = matrix(rnorm(n_test*p), ncol=p)
    Z = z_true(Xt,n_test)
    yt = gen_true(Xt,Z)
}
if(model_type == "reg_simple"){
    
    bt0 = seq(-2,2,length.out=p)
    
    gen_true = function(X,Z){
        bt0 = seq(-2,2,length.out=p)
        y = X%*%bt0  + Z
        y = matrix(y,length(y),1)
        return(y)
    }
    
    z_true = function(x,n){zz = rnorm(n, mean=0, sd=sigma0);return(zz)}
    
    #Seed = 128783
    set.seed(Seed)
    X = matrix(rnorm(n*p), ncol=p)
    Z = z_true(X,n)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<=5000){n_test=0.2*n}
    if(n>5000&n<=10000){n_test=500}
    Xt = matrix(rnorm(n_test*p), ncol=p)
    Xt[1,] = c(1, rep(0,p-1))
    Xt[2,] = rep(0, p)
    Xt[3,] = c(rep(0,p-1),1)
    Z = rnorm(n_test, mean=0, sd=sigma0)
    yt = gen_true(Xt,Z)
}
if(model_type == "reg_nonconstant"){
    
    bt0 = seq(-2,2,length.out=p)
    
    gen_true = function(X,Z){
        bt0 = seq(-2,2,length.out=p)
        y = X%*%bt0 + Z
        y = matrix(y,length(y),1)
        return(y)
    }
    
    z_true = function(x,n){
        #U=runif(n);zz=rnorm(n, mean=0, sd=sigma0)*(0.5*(U<0.5)+1.0*(U>=0.5))
        zz = rnorm(n, mean=0, sd=sigma0)*(0.6*(x[,1]<0)+1.2*(x[,1]>=0))
        return(zz)}
    
    set.seed(128783)
    X = matrix(rnorm(n*p), ncol=p)
    Z = z_true(X,n)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<=5000){n_test=0.1*n}
    if(n>5000&n<=10000){n_test=500}
    Xt = matrix(rnorm(n_test*p), ncol=p)
    Xt[1,] = c(1, rep(0,p-1))
    Xt[2,] = c(0, 1, rep(0,p-2))
    Xt[3,] = c(0, 0, 1, rep(0, p-3))
    Z = z_true(Xt, n_test)
    yt = gen_true(Xt,Z)
}
if(model_type == "reg_skew"){
    
    
    bt0 = seq(-2,2,length.out=p)
    
    gen_true = function(X,Z){
        bt0 = seq(-2,2,length.out=p)
        y = X%*%bt0  + Z
        y = matrix(y,length(y),1)
        return(y)
    }
    
    z_true = function(x,n){zz = rchisq(n, df=1, ncp=1)*(x[,1]>=0.5) + log(rchisq(n, df=1, ncp=1))*(x[,1]<0.5);return(zz)}
    
    Seed = 128783
    set.seed(Seed)
    X = matrix(rnorm(n*p), ncol=p)
    Z = z_true(X,n)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<=5000){n_test=0.1*n}
    if(n>5000&n<=10000){n_test=500}
    Xt = matrix(rnorm(n_test*p), ncol=p)
    Xt[1,] = c(1, rep(0,p-1))
    Xt[2,] = c(0, 1, rep(0,p-2))
    Xt[3,] = c(0, 0, 1, rep(0, p-3))
    Z = z_true(Xt,n_test)
    yt = gen_true(Xt,Z)
}
if(model_type == "reg_linear"){
    
    
    bt0 = seq(-2,2,length.out=p)
    
    gen_true = function(X,Z){
        bt0 = seq(-2,2,length.out=p)
        y = X%*%bt0  + Z
        y = matrix(y,length(y),1)
        return(y)
    }
    
    z_true = function(x,n){zz = rnorm(n,0,sigma0);return(zz)}
    
    Seed = 128783
    set.seed(Seed)
    X = matrix(rnorm(n*p), ncol=p)
    Z = z_true(X,n)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<=5000){n_test=0.1*n}
    if(n>5000&n<=10000){n_test=500}
    Xt = matrix(rnorm(n_test*p), ncol=p)
    Z = z_true(Xt,n_test)
    yt = gen_true(Xt,Z)
}
if(model_type == "reg_norm"){
    
    
    bt0 = seq(-2,2,length.out=p)
    
    gen_true = function(X,Z){
        bt0 = seq(-2,2,length.out=p)
        y = X%*%bt0  + Z
        y = matrix(y,length(y),1)
        return(y)
    }
    
    z_true = function(x,n){
        sdd = apply(x, 1, function(x) sum(abs(x)))
        zz = rnorm(n, 0, sd=exp(0.5*sdd));return(zz)}
    
    Seed = 128783
    set.seed(Seed)
    X = matrix(runif(n*p, -1, 1), ncol=p)
    Z = z_true(X,n)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<=5000){n_test=0.1*n}
    if(n>5000&n<=10000){n_test=500}
    Xt = matrix(rnorm(n_test*p), ncol=p)
    Xt[1,] = c(1, rep(0,p-1))
    Xt[2,] = c(0, 1, rep(0,p-2))
    Xt[3,] = c(0, 0, 1, rep(0, p-3))
    Z = z_true(Xt,n_test)
    yt = gen_true(Xt,Z)
}
if(model_type == "reg_skewV2"){
    
    
    bt0 = seq(-2,2,length.out=p)
    
    gen_true = function(X,Z){
        bt0 = seq(-2,2,length.out=p)
        y = X%*%bt0  + Z
        y = matrix(y,length(y),1)
        return(y)
    }
    
    z_true = function(x,n){zz = HyperbolicDist::rskewlap(n,c(1.5,1,0))*(x[,2]>=0.25)+rnorm(n)**(x[,2]<0.25);return(zz)}
    
    #Seed = 128783
    set.seed(Seed)
    X = matrix(rnorm(n*p), ncol=p)
    Z = z_true(X,n)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<=5000){n_test=0.1*n}
    if(n>5000&n<=10000){n_test=500}
    Xt = matrix(rnorm(n_test*p), ncol=p)
    Xt[1,] = c(1, rep(0,p-1))
    Xt[2,] = c(0, 1, rep(0,p-2))
    Xt[3,] = c(0, 0, 1, rep(0, p-3))
    Z = z_true(Xt,n_test)
    yt = gen_true(Xt,Z)
}
if(model_type == "reg_skewV3"){
    
    
    bt0 = seq(-2,2,length.out=p)
    
    gen_true = function(X,Z){
        bt0 = seq(-2,2,length.out=p)
        y = X%*%bt0  + Z
        y = matrix(y,length(y),1)
        return(y)
    }
    
    z_true = function(x,n){zz = log(rchisq(n, df=1, ncp=1))*(x[,2]>=0.25)+rnorm(n)**(x[,2]<0.25);return(zz)}
    
    #Seed = 128783
    set.seed(Seed)
    X = matrix(rnorm(n*p), ncol=p)
    Z = z_true(X,n)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<=5000){n_test=0.1*n}
    if(n>5000&n<=10000){n_test=500}
    Xt = matrix(rnorm(n_test*p), ncol=p)
    Xt[1,] = c(1, rep(0,p-1))
    Xt[2,] = c(0, 1, rep(0,p-2))
    Xt[3,] = c(0, 0, 1, rep(0, p-3))
    Z = z_true(Xt,n_test)
    yt = gen_true(Xt,Z)
}
if(model_type == "reg_multimode"){
    
    
    bt0 = seq(-2,2,length.out=p)
    
    gen_true = function(X,Z){
        n0 = nrow(X)
        p = ncol(X)
        
        bt0 = seq(-2,2,length.out=p-1)
        #y = X%*%bt0  + Z
        y = base::sample(c(-2,2),size=n0,replace=T)*X[,1]+X[,2:p]%*%bt0+Z
        y = matrix(y,length(y),1)
        return(y)
    }
    
    z_true = function(x,n){zz = rnorm(n, mean=0, sd=sigma0);return(zz)}
    
    #Seed = 128783
    set.seed(Seed)
    X = matrix(rnorm(n*p), ncol=p)
    Z = z_true(X,n)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<=5000){n_test=0.1*n}
    if(n>5000&n<=10000){n_test=500}
    Xt = matrix(rnorm(n_test*p), ncol=p)
    Xt[1,] = c(1, rep(0,p-1))
    Xt[2,] = c(0, 1, rep(0,p-2))
    Xt[3,] = c(0, 0, 1, rep(0, p-3))
    Z = z_true(Xt,n_test)
    yt = gen_true(Xt,Z)
}
if(model_type == "reg_heavy"){
    
    
    bt0 = seq(-2,2,length.out=p)
    
    gen_true = function(X,Z){
        bt0 = seq(-2,2,length.out=p)
        y = X%*%bt0  + Z
        y = matrix(y,length(y),1)
        return(y)
    }
    
    z_true = function(x,n){n0 = n;zz = rt(n0, df=4)*(x[,2]<=0.5)+rlnorm(n0,0,1)*(x[,2]>0.5);return(zz)}
    
    set.seed(128783)
    X = matrix(rnorm(n*p), ncol=p)
    Z = rchisq(n, df=2, ncp=1)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<=5000){n_test=0.1*n}
    if(n>5000&n<=10000){n_test=500}
    Xt = matrix(rnorm(n_test*p), ncol=p)
    Xt[1,] = c(1, rep(0,p-1))
    Xt[2,] = c(0, 1, rep(0,p-2))
    Xt[3,] = c(0, 0, 1, rep(0, p-3))
    Z = z_true(Xt,n_test)
    yt = gen_true(Xt,Z)
}
if(model_type == "M3"){
    
    
    bt0 = seq(-2,2,length.out=p)
    
    gen_true = function(X,Z){
        y = (5 + X[,1]^2/3 + X[,2]^2 + X[,3]^2 + X[,4] + X[,5])* exp(0.5*Z)
        y = matrix(y,length(y),1)
        return(y)
    }
    
    z_true = function(x,n){U=runif(n);zz=rnorm(n,-2,1)*(U<0.5)+rnorm(n,2,1)*(U>=0.5);return(zz)}
    
    set.seed(128783)
    X = matrix(rnorm(n*p), ncol=p)
    Z = z_true(X,n)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<=5000){n_test=0.1*n}
    if(n>5000&n<=10000){n_test=500}
    Xt = matrix(rnorm(n_test*p), ncol=p)
    Xt[1,] = c(1, rep(0,p-1))
    Xt[2,] = c(0, 1, rep(0,p-2))
    Xt[3,] = c(0, 0, 1, rep(0, p-3))
    Z =z_true(Xt,n_test)
    yt = gen_true(Xt,Z)
}
if(model_type == "M2"){
    
    
    bt0 = seq(-2,2,length.out=p)
    
    gen_true = function(X,Z){
        y = (X[,1]^2+exp(X[,2]+X[,3]/3)+X[,4]-X[,5])+(0.5+X[,2]^2/2+X[,5]^2/2)*Z
        y = matrix(y,length(y),1)
        return(y)
    }
    
    z_true = function(x,n){zz=rnorm(n);return(zz)}
    
    set.seed(128783)
    X = matrix(rnorm(n*p), ncol=p)
    Z = z_true(X,n)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<=5000){n_test=0.1*n}
    if(n>5000&n<=10000){n_test=500}
    Xt = matrix(rnorm(n_test*p), ncol=p)
    #Xt[1,] = c(1, rep(0,p-1))
    #Xt[2,] = c(0, 1, rep(0,p-2))
    #Xt[3,] = c(0, 0, 1, rep(0, p-3))
    Z = z_true(Xt,n_test)
    yt = gen_true(Xt,Z)
}
if(model_type == "reg_multi_y"){
    
    
    #Non-linear
    #gen_true = function(X, Z){
    #    y = matrix(0, nrow=nrow(X), ncol=2)
    #    y[,1] = sin(X) + Z[,1]
    #    y[,2] = log(X^2) + Z[,2]
    #    return(y)
    #}
    #Linear
    p = 1
    gen_true = function(X, Z){
        y = matrix(0, nrow=nrow(X), ncol=2)
        y[,1] = X + Z[,1]
        y[,2] = X + Z[,2]
        return(y)
    }
    
    #z_true = function(x,n){zz=cbind(rnorm(n,0,1.5), rnorm(n,0,1.2));return(zz)}
    z_true = function(x,n){r=0.9;mu0=rep(0,2);Sigma0=matrix(c(1,r,r,1),2,2);zz=MASS::mvrnorm(n,mu0,Sigma0);return(zz)}
    
    set.seed(128783)
    X = matrix(rnorm(n*p), n, p)
    Z = z_true(X,n)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<=5000){n_test=0.1*n}
    if(n>5000&n<=10000){n_test=500}
    Xt = matrix(rnorm(n_test*p), ncol=p)
    #Xt[1,] = c(1, rep(0,p-1))
    #Xt[2,] = c(0, 1, rep(0,p-2))
    #Xt[3,] = c(0, 0, 1, rep(0, p-3))
    Z = z_true(Xt, n_test)
    yt = gen_true(Xt, Z)
}
if(model_type == "two_moon"){
    
    
    
    gen_true = function(X, Z){
        y = matrix(0, nrow=nrow(X), ncol=2)
        sigma00 = 0.1
        for(i in 1:nrow(X)){
            a= runif(1, min=0, max=pi)
            if(X[i]==1){y[i,] = c(cos(a)+0.5+rnorm(1,0,sigma00),sin(a)-1/6+rnorm(1,0,sigma00))}
            if(X[i]==2){y[i,] = c(cos(a)-0.5+rnorm(1,0,sigma00),-sin(a)+1/6+rnorm(1,0,sigma00))}
        }
        return(y)
    }
    
    z_true = function(x,n){zz=cbind(rnorm(n,0,0.2), rnorm(n,0,0.2));return(zz)}
    
    set.seed(128783)
    X = matrix(sample(c(1,2),size=n*p,replace=T), n, p)
    Z = z_true(X,n)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<=5000){n_test=0.1*n}
    if(n>5000&n<=10000){n_test=500}
    Xt = matrix(sample(c(1,2),size=n_test*p,replace=T), n_test, p)
    #Xt[1,] = c(1, rep(0,p-1))
    #Xt[2,] = c(0, 1, rep(0,p-2))
    #Xt[3,] = c(0, 0, 1, rep(0, p-3))
    Z = z_true(Xt, n_test)
    yt = gen_true(Xt, Z)
}
if(model_type == "bi_sin"){
    
    
    p = 1
    gen_true = function(X, Z){
        y = matrix(0, nrow=nrow(X), ncol=2)
        U = seq(0, 2*pi, length=nrow(X)*p)
        y[,2] = sin(U) + Z[,1] + X
        y[,1] = U + Z[,2]
        return(y)
    }
    
    z_true = function(x,n){zz=cbind(rnorm(n,0,0.3), rnorm(n,0,0.3));return(zz)}
    
    set.seed(128783)
    #X = matrix(sample(c(1,2),size=n*p,replace=T), n, p)
    X = matrix(1,n,p)
    Z = z_true(X,n)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<=5000){n_test=0.1*n}
    if(n>5000&n<=10000){n_test=500}
    #Xt = matrix(sample(c(1,2),size=n_test*p,replace=T), n_test, p)
    Xt = matrix(1, n_test, p)
    #Xt[1,] = c(1, rep(0,p-1))
    #Xt[2,] = c(0, 1, rep(0,p-2))
    #Xt[3,] = c(0, 0, 1, rep(0, p-3))
    Z = z_true(Xt, n_test)
    yt = gen_true(Xt, Z)
}
if(model_type == "bi_laplace"){
    
    library("LaplacesDemon")
    p = 1
    gen_true = function(X, Z){
        y = matrix(0, nrow=nrow(X), ncol=2)
        #U = seq(0, 2*pi, length=nrow(X)*p)
        y[,2] = 1 * (X >= 0.5) + Z[,1]
        y[,1] = (-1) * (X < 0.5) + Z[,2]
        return(y)
    }
    
    z_true = function(x,n){mu0=rep(0,2);Sigma0=matrix(c(0.5,-0.4,-0.4,0.5),2,2);zz=rmvl(n,mu0,Sigma0);return(zz)}
    
    set.seed(128783)
    X = matrix(runif(n*p), n, p)
    #X = matrix(sample(c(1,2),size=n*p,replace=T), n, p)
    #X = matrix(1,n,p)
    Z = z_true(X,n)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<=5000){n_test=0.1*n}
    if(n>5000&n<=10000){n_test=500}
    Xt = matrix(runif(n_test*p), n_test, p)
    #Xt = matrix(sample(c(1,2),size=n_test*p,replace=T), n_test, p)
    #Xt = matrix(1, n_test, p)
    #Xt[1,] = c(1, rep(0,p-1))
    #Xt[2,] = c(0, 1, rep(0,p-2))
    #Xt[3,] = c(0, 0, 1, rep(0, p-3))
    Z = z_true(Xt, n_test)
    yt = gen_true(Xt, Z)
}
if(model_type == "bi_normmix"){
    
    
    #Non-linear
    #gen_true = function(X, Z){
    #    y = matrix(0, nrow=nrow(X), ncol=2)
    #    y[,1] = sin(X) + Z[,1]
    #    y[,2] = log(X^2) + Z[,2]
    #    return(y)
    #}
    #Linear
    p = 1
    gen_true = function(X, Z){
        y = matrix(0, nrow=nrow(X), ncol=2)
        y[,1] = X + Z[,1]
        y[,2] = X + Z[,2]
        return(y)
    }
    
    #z_true = function(x,n){zz=cbind(rnorm(n,0,1.5), rnorm(n,0,1.2));return(zz)}
    z_true = function(x,n){
        zz = matrix(0, nrow=nrow(x), ncol=2)
        for(i in 1:nrow(x)){
            U = runif(1)
            if(U >= 0.5 ){r = 0.45;mu0=c(-0.5,0)}
            if(U < 0.5){r = -0.45;mu0=c(0.5,0)}
            Sigma0=matrix(c(0.5,r,r,0.5),2,2);
            zz[i,]=MASS::mvrnorm(1,mu0,Sigma0)}
        return(zz)
    }
    
    set.seed(128783)
    X = matrix(rnorm(n*p), n, p)
    Z = z_true(X,n)
    y = gen_true(X,Z)
    
    n_test = 0.05/2*n
    if(n<=5000){n_test=0.1*n}
    if(n>5000&n<=10000){n_test=500}
    Xt = matrix(rnorm(n_test*p), ncol=p)
    #Xt[1,] = c(1, rep(0,p-1))
    #Xt[2,] = c(0, 1, rep(0,p-2))
    #Xt[3,] = c(0, 0, 1, rep(0, p-3))
    Z = z_true(Xt, n_test)
    yt = gen_true(Xt, Z)
}
