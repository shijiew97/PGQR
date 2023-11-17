#### Useful Function ####
rm(list=ls())
findThresholdHPD = function(binSize,estimates,confidence){
    estimates=as.vector(estimates)
    maxDensity=max(estimates)
    minDensity=min(estimates)
    newCut=(maxDensity+minDensity)/2
    eps=1
    ii=1
    while(ii<=1000)
    {
        prob=sum(binSize*estimates*(estimates>newCut))
        eps=abs(confidence-prob)
        if(eps<0.0000001) break; # level found
        if(confidence>prob) maxDensity=newCut
        if(confidence<prob) minDensity=newCut
        newCut=(maxDensity+minDensity)/2
        ii=ii+1
    }
    return(newCut)
}
flex_eval = function(X,y,Xt,yt,type="NN"){
    
    ## Split the dataset
    library('FlexCoDE')
    n = nrow(X)
    n_train = 0.7*n
    random_ind = sample(1:n)
    X_train = X[random_ind[1:n_train],];y_train = y[random_ind[1:n_train],]
    X_val = X[random_ind[(n_train+1):n],];y_val = y[random_ind[(n_train+1):n],]
    
    ## Fit the training data
    if(type=="NN"){fit = fitFlexCoDE(X_train,y_train,X_val,y_val,Xt,yt,nIMax=20,regressionFunction=regressionFunction.NN, n_grid =500)}
    if(type=="SPAM"){fit = fitFlexCoDE(X_train,y_train,X_val,y_val,Xt,yt,nIMax=2,regressionFunction= regressionFunction.SpAM, n_grid =500)}
    if(type=="XGB"){fit = FlexZBoost(X_train,y_train,X_val,y_val,Xt,yt,nIMax=20,n_grid=500)}
    if(type=="Series"){fit = fitFlexCoDE(X_train,y_train,X_val,y_val,Xt,yt,nIMax=20,regressionFunction= regressionFunction.Series, n_grid =500)}
    ## Get the predition cde and threshold for prediction band
    predictedValues = predict(fit,Xt,B=500,predictionBandProb=0.95)
    cde = predictedValues$CDE
    thres = predictedValues$th
    
    ## Calculate the statistics: mean;quantile;using sample code online
    ## Numerical intergation
    x=predictedValues$z;dx = diff(x);wd = mean(dx)
    flex_stat = array(0, dim=c(9,length(yt)))
    for(i in 1:nrow(Xt)){
        ht = cde[i,]
        #mean and sd
        flex_stat[1,i] = sum(x*wd*ht)
        flex_stat[2,i] = sqrt(sum(x^2*ht*wd)-(sum(x*ht*wd))^2)
        #quantile
        flex_stat[3:9,i]=c(min(x[cumsum(ht*wd)>=.05]),min(x[cumsum(ht*wd)>=.1]),min(x[cumsum(ht*wd)>=.25]),min(x[cumsum(ht*wd)>=.5]),min(x[cumsum(ht*wd)>=.75]),min(x[cumsum(ht*wd)>=.90]),min(x[cumsum(ht*wd)>=.95]))
    }
    return(list(x, cde, flex_stat, thres))
}
NNK_eval = function(X,y,Xt,yt){
    
    ## Split the dataset: 70% training
    library("NNKCDE")
    n = nrow(X)
    n_train = 0.7*n
    random_ind = sample(1:n)
    X_train = X[random_ind[1:n_train],];y_train = y[random_ind[1:n_train],]
    X_val = X[random_ind[(n_train+1):n],];y_val = y[random_ind[(n_train+1):n],]
    
    ## Fitting the training data: sample code online
    grid = flex_eval(X,y,Xt,yt,"NN")[[1]]
    Bin_size = (max(grid)-min(grid))/length(grid)
    fit = NNKCDE$new(X_train,y_train)
    fit$tune(X_val, y_val, k_grid = c(5,10,100,1000), h_grid=c(0.05,0.1,0.5))
    
    ## Get the prediction cde and threshold
    cde=fit$predict(Xt,grid)
    
    ## Calculate the statistics: mean, sd and quantile difference
    x=grid;dx = diff(x);wd = mean(dx)
    stat = array(0, dim=c(9,length(yt)))
    thres = array(0, dim=c(nrow(Xt),1))
    for(i in 1:nrow(Xt)){
        ht = cde[i,]
        #mean and sd
        stat[1,i] = sum(x*wd*ht)
        stat[2,i] = sqrt(sum(x^2*ht*wd)-(sum(x*ht*wd))^2)
        #quantile
        stat[3:9,i]=c(min(x[cumsum(ht*wd)>=.05]),min(x[cumsum(ht*wd)>=.1]),min(x[cumsum(ht*wd)>=.25]),min(x[cumsum(ht*wd)>=.5]),min(x[cumsum(ht*wd)>=.75]),min(x[cumsum(ht*wd)>=.90]),min(x[cumsum(ht*wd)>=.95]))
        thres[i,] = findThresholdHPD(Bin_size, cde[i,], 0.95)
    }
    
    ## return the statistics
    return(list(x,cde,stat,thres))
}
HDII = function(x, density, coverage){
    samples = sample(x, size=5000, prob=density, replace=T)
    CI = quantile(samples, c(0.025,0.975))
    return(CI)
}
RF_eval = function(X,y,Xt,yt){
    
    ##No need to split the dataset
    library(RFCDE)
    ## Fitting the training data: sample code online
    grid = flex_eval(X,y,Xt,yt,"NN")[[1]]
    Bin_size = (max(grid)-min(grid))/length(grid)
    
    ##Initilization online provided
    n_trees <- 1000 # Number of trees in the forest
    mtry <- 4 # Number of variables to potentially split at in each node
    node_size <- 20 # Smallest node size
    n_basis <- 15 # Number of basis functions
    bandwidth <- 0.2
    
    ##Fitting
    forest = RFCDE::RFCDE(X, y, n_trees = n_trees, mtry = mtry,
                          node_size = node_size, n_basis = n_basis)
    
    ##Get the cde
    cde = predict(forest, Xt, grid, response = "CDE", bandwidth = bandwidth)
    
    ## Calculate the statistics: mean, sd and quantile difference
    x=grid;dx = diff(x);wd = mean(dx)
    stat = array(0, dim=c(9,length(yt)))
    thres = array(0, dim=c(nrow(Xt),1))
    for(i in 1:nrow(Xt)){
        ht = cde[i,]
        #mean and sd
        stat[1,i] = sum(x*wd*ht)
        stat[2,i] = sqrt(sum(x^2*ht*wd)-(sum(x*ht*wd))^2)
        #quantile
        stat[3:9,i]=c(min(x[cumsum(ht*wd)>=.05]),min(x[cumsum(ht*wd)>=.1]),min(x[cumsum(ht*wd)>=.25]),min(x[cumsum(ht*wd)>=.5]),min(x[cumsum(ht*wd)>=.75]),min(x[cumsum(ht*wd)>=.90]),min(x[cumsum(ht*wd)>=.95]))
        thres[i,] = findThresholdHPD(Bin_size, cde[i,], 0.95)
    }
    
    ## return the statistics
    return(list(x,cde,stat,thres))
}


#### Setting the working directory ####
#rm(list=ls())
#setwd("~/Dropbox/Shijie/GR")
#setwd("C:/Users/18036/Dropbox/Shijie/GR")

#### Simulation setting ####
#model_type = "reg_P111"       #Simualtion 1
#model_type = "reg_nonparam"   #Simualtion 4
#model_type = "reg_simple"     #Overfitting illustration
#model_type = "reg_skew"       #Simulation 2
#model_type = "reg_linear"     #Simulation 5 (Small variance)
#model_type = "reg_multimode"  #Simulation 3
#model_type = "reg_norm"       #Simulation 6

#### Different model ####
#method = "QR_m"               #PGQR
#method = "QR_nopen_m"         #GQR
#method = "fGAN"               #GCDS
#method = "fGAN_C"             #deep-GCDS
#method = "WGAN"               #WGCS


#### Calculate the MSE of median, mean, sd, quantile ##
evals = c("E(Y|X)","sd(Y|X)","10%","20%","30%","40%","50%","60%","70%","80%","90%","cov","width", "TV", "HD", "KL")
method0 = c("fGAN","fGAN_C", "WGAN", "QR_m", "flexcode_NN","flexcode_SPAM","flexcode_XGB","RFCDE")
res_table = matrix(0, nrow=length(method0), ncol=length(evals))
colnames(res_table) = evals
rownames(res_table) = method0
num_rep = 1*20                 #number of replication
n_test = 100                   #number of out-of-sample test points
tau_cand = seq(0.1, 0.9, length=9)

#### Here: choose different alpha value ####
fac = 1.0*1                    #alpha value
n = 2000                       #sample size
p = 5.0                        #covariate dimension
stat_sd = array(0, dim=c(num_rep, 4))
target = 4

#### Begin evulation of performance ####
#### Specially for Generative models ####
for(mm in target){
#for(mm in 1:4){
    #mm = 1
    library("HDInterval")
    method = method0[mm]
    print(paste(method, " evaluation starts!"))
    
    para_int = 1
    n = 2000
    if(para_int == 1){
        N = 1
        m = 1
        test = 1
        gpu_ind = 0
        verb = 1
        tsne_fit = 0
        boot_size = 1000
        l1pen = 0
        Seed = 128783
        
        lam_min = -30.0
        lam_max = 0.1
        
        
        #n = 2000
        n0 = n
        zn = 100
        ntest = 100
        #p = 10/2
        S = n
        #fac = 1.0
        if(method == "QR"){fac = 10.0}
        
        
        L = 3
        batchnorm_on = 1
        num_it = 10000
        hidden_size = 1000
        lr = 0.0001
        lr_power = 0.2
        lrdecay = 1
        k = 100
        NN_type = "MLP"
        
        sigma0 = 1
    }
    p = 5.0
    if(model_type == "reg_P111"){p=1}
    if(model_type == "reg_linear"){p=1;sigma0=0.1}
    
    pen_on = 0
    if(method == "QR"){pen_on = 1}
    if(method == "QR_m"){pen_on = 1}
    if(method == "fGAN"){num_it=10000*5}
    if(method == "WGAN"){num_it=10000*5}
    
    if(pen_on == 0){data_type = paste(model_type,"_",method,sep="")}
    if(pen_on == 1){data_type = paste(model_type,"_",method,"_pen",sep="")}
    
    if(method == "QR_m"){
        data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K", "_COMPARISON_", "a_",fac, sep="")
    }else{
        data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K", "_COMPARISON", sep="")
    }
    
    path0 = paste("/result/",n,"/",sep="")
    read_path = paste(getwd(), path0, data_type, "_gr.RData", sep="")
    ymat_pen_opt = readRDS(read_path)#(10,1000,200)
    
    for(j in 1:num_rep){
        #if(model_type != "reg_multimode"){Seed = Seed + j -1}
        source("./R_code/data_gen.R")
        n_test = 100
        for(i in 101:200){
            
            yt_gen_pen = ymat_pen_opt[j,,i]
            
            if(method == "QR_m"){
                quant = quantile(yt_gen_pen, probs=c(0.02,0.98))
                ind_out = c(which(yt_gen_pen>=quant[2]), which(yt_gen_pen<=quant[1]))
                yt_gen_pen = yt_gen_pen[-ind_out]
                n0 = length(yt_gen_pen)
            }
            
            X_gen = matrix(rep(Xt[i,], each=n0), n0, p)
            Z_gen = z_true(X_gen,n0)
            y_gen = gen_true(X_gen, Z_gen)
            
            stat_gen = c(mean(yt_gen_pen),sd(yt_gen_pen),quantile(yt_gen_pen,probs=tau_cand))
            stat_true = c(mean(y_gen),sd(y_gen),quantile(y_gen,probs=tau_cand))
            
            ##Total deviance of quantile function
            rk = c(0)
            for(k in 1:9){
                cur = sum( (y_gen <= stat_gen[k+2]) )/n0
                rk = c(rk, cur)
            }
            rk = c(rk, 1)
            pk = diff(rk)
            tv = sum(abs(0.1 - pk))
            hd = sum( (sqrt(0.1)-sqrt(pk))^2 )
            kl = -sum( 0.1*log(pk) )
            
            res_table[mm, length(evals)-2] = res_table[mm, length(evals)-2] + tv
            res_table[mm, length(evals)-1] = res_table[mm, length(evals)-1] + hd
            res_table[mm, length(evals)] = res_table[mm, length(evals)] + kl
            
            num_m = length(evals)-5
            res_table[mm,1:num_m] = res_table[mm,1:num_m] + (stat_true-stat_gen)^2
            
            stat_sd[j, 1:2] = stat_sd[j, 1:2] + ((stat_true-stat_gen)^2)[1:2]/n_test
            stat_sd[j, 3] = stat_sd[j, 3] + tv/n_test
            stat_sd[j, 4] = stat_sd[j, 3] + hd/n_test
            
            CI_pen = quantile(yt_gen_pen, probs=c(0.025,0.975))#hdi(yt_gen_pen, credMass=0.95)
            if(length(CI_pen) == 2){
                if(yt[i]>=CI_pen[1]&yt[i]<CI_pen[2]){
                    res_table[mm,num_m+1]=res_table[mm,num_m+1]+1}
                res_table[mm,num_m+2]=res_table[mm,num_m+2]+abs(CI_pen[1]-CI_pen[2])}
            else{
                if(yt[i]>=CI_pen[1]&yt[i]<CI_pen[2]){res_table[mm,num_m+1]=res_table[mm,num_m+1]+1}
                if(yt[i]>=CI_pen[3]&yt[i]<CI_pen[4]){res_table[mm,num_m+1]=res_table[mm,num_m+1]+1}
                res_table[mm,num_m+2]=res_table[mm,num_m+2]+abs(CI_pen[1]-CI_pen[2]+CI_pen[3]-CI_pen[4])}
        }
    }
}

#res_table = round(res_table/(num_rep*n_test),4)
#print(res_table[target, c(1,2,12,13,14,15,16)])
#print(apply(stat_sd, 2, sd))


#### Tranditional CDE models ####
type0 = "NN"
#type0 = "SPAM"
#type0 = "XGB"
#type0 = "RFCDE"
for(type0 in c("NN", "SPAM", "XGB","RFCDE")){
    for(j in 1:num_rep){
        
        print(paste(type0, ": iteration <<", j, ">> starts!", sep=""))
        para_int = 1
        if(para_int == 1){
            N = 1
            m = 1
            test = 1
            gpu_ind = 0
            verb = 1
            tsne_fit = 0
            boot_size = 1000
            l1pen = 0
            Seed = 128783
            
            lam_min = -30.0
            lam_max = 4.0
            
            
            n = 2000
            n0 = n
            zn = 100
            ntest = 100
            S = n
            fac = 1.0
            if(method == "QR"){fac = 10.0}
            
            
            L = 3
            batchnorm_on = 1
            num_it = 10000
            hidden_size = 1500
            lr = 0.0001
            lr_power = 0.2
            lrdecay = 1
            k = 100
            NN_type = "MLP"
            
            sigma0 = 1
        }
        
        p = 5.0
        if(model_type == "reg_P111"){p=1}
        if(model_type == "reg_linear"){p=1;sigma0=0.1}
        
        if(model_type == "reg_multimode" | model_type == "reg_linear"){Seed = Seed}
        else{Seed = Seed + j -1}
        source("./R_code/data_gen.R")
        
        
        #plot(flex_grid, flex_cde[i,]);abline(h=flex_thres[i,])
        if(type0 == "NN"){flex_fit = flex_eval(X,y,Xt,yt,type0);cde = flex_fit[[2]];grid = flex_fit[[1]];thres = flex_fit[[4]];mm = 5}
        if(type0 == "SPAM"){flex_fit = flex_eval(X,y,Xt,yt,type0);cde = flex_fit[[2]];grid = flex_fit[[1]];thres = flex_fit[[4]];mm = 6}
        if(type0 == "XGB"){flex_fit = flex_eval(X,y,Xt,yt,type0);cde = flex_fit[[2]];grid = flex_fit[[1]];thres = flex_fit[[4]];mm = 7}
        if(type0 == "RFCDE"){flex_fit = RF_eval(X,y,Xt,yt);cde = flex_fit[[2]];grid = flex_fit[[1]];thres = flex_fit[[4]];mm = 8}
        
        n_test = 100
        for(i in 101:200){
            X_gen = matrix(rep(Xt[i,], each=n0), n0, p)
            Z_gen = z_true(X_gen,n0)
            y_gen = gen_true(X_gen, Z_gen)
            
            num_m = length(evals)-3
            stat_true = c(mean(y_gen),sd(y_gen),quantile(y_gen,probs=c(0.05,0.1,0.25,0.5,0.75,0.9,0.95)))
            res_table[mm,1:num_m] = res_table[mm,1:num_m] + (stat_true-flex_fit[[3]][,i])^2
            
            stat_sd[j, 1:2] = stat_sd[j, 1:2] + ((stat_true-flex_fit[[3]][,i])^2)[1:2]/n_test
            
            #get CI
            rr = cde[i,]-thres[i,]+1e-20
            rr1 = rr[-1];rr1[length(rr)] = rr[length(rr)]
            r = rr*rr1
            root_ind = which(r<=0)+1
            
            CI_pen = grid[root_ind]
            #if(type0 == "NNKCDE"){CI_pen = HDII(grid,cde[i,],0.95)}
            if(model_type=="reg_P111"){CI_pen = HDII(flex_fit[[1]], cde[i,], 0.95)}
            if(model_type=="reg_linear"){CI_pen = HDII(flex_fit[[1]], cde[i,], 0.95)}
            if(length(CI_pen)!=0){
                if(length(CI_pen) == 1){
                    CI_pen0 = c(CI_pen, grid[1])
                    CI_pen00 = c(CI_pen, grid[length(r)])
                    if(diff(CI_pen0)>=diff(CI_pen00)){CI_pen = CI_pen0}
                    if(diff(CI_pen0)<diff(CI_pen00)){CI_pen = CI_pen00}
                    
                }
                if(length(CI_pen)%%2 !=0){
                    max_ind = which.max(diff(CI_pen))
                    CI_pen = CI_pen[max_ind:(max_ind+1)]}
                if(length(CI_pen) == 2){
                    if(yt[i]>=CI_pen[1]&yt[i]<CI_pen[2]){
                        res_table[mm,num_m+1]=res_table[mm,num_m+1]+1}
                    res_table[mm,num_m+2]=res_table[mm,num_m+2]+abs(CI_pen[1]-CI_pen[2])}
                else{
                    if(yt[i]>=CI_pen[1]&yt[i]<CI_pen[2]){res_table[mm,num_m+1]=res_table[mm,num_m+1]+1}
                    if(yt[i]>=CI_pen[3]&yt[i]<CI_pen[4]){res_table[mm,num_m+1]=res_table[mm,num_m+1]+1}
                    res_table[mm,num_m+2]=res_table[mm,num_m+2]+abs(CI_pen[1]-CI_pen[2]+CI_pen[3]-CI_pen[4])}}
            if(length(CI_pen)==0){res_table[mm,num_m+2]=res_table[mm,num_m+2]+mean(res_table[mm,num_m+2])}
            
        }
    }
}

#### Print the table results ####
res_table = round(res_table/(num_rep*n_test),4)
#print(apply(stat_sd, 2, sd))
print(res_table)

#### Get the saving path ####
if(pen_on == 0){data_type = paste(model_type,"_",method,sep="")}
if(pen_on == 1){data_type = paste(model_type,"_",method,"_pen",sep="")}

data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K", "_TABLE", sep="")

#### Save the results ####
path0 = paste("/result/",n,"/",sep="")
save_path = paste(getwd(), path0, data_type, "_gr.RData", sep="")
saveRDS(res_table, save_path)



