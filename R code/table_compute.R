#### Setting the working directory ####
#rm(list=ls())
#setwd("~/Dropbox/Shijie/GR")

#### Simulation setting ####
model_type = "reg_P111"        #Simualtion 1
#model_type = "reg_nonparam"   #Simualtion 4
#model_type = "reg_simple"     #Overfitting illustration
#model_type = "reg_skew"       #Simulation 2
#model_type = "reg_linear"     #Small variance
#model_type = "reg_multimode"  #Simulation 3


#### Different model ####
method = "QR_m"                #PGQR
#method = "QR_nopen_m"         #GQR
#method = "fGAN"               #GCDS
#method = "fGAN_C"             #deep-GCDS
#method = "WGAN"               #WGCS



#### Save the result ##
para_int = 1                   #whether to initize the parameter
num_rep = 1*20                 #number of replication
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
    
    
    n = 2000
    n0 = n
    zn = 100
    ntest = 100
    p = 10/2
    S = n
    fac = 1.0
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
res = array(0, dim=c(num_rep,boot_size,0.1*n))

#### Evaluation of replication ##
for(kk in 1:num_rep){
    
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
        
        
        n = 2000
        n0 = n
        zn = 100
        ntest = 100
        p = 10/2
        S = n
        fac = 1.0
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
    #if(model_type != "reg_multimode"){Seed = Seed + kk -1}
    #### Get the data generation ####
    source("./R code/data_gen.R")
    
    #### Fit the PGQR model ####
    if(method == "fGAN"){L=2;hidden_size=40;zn=3;pen_on=0}
    if(method == "fGAN_C"){L=3;hidden_size=1000;zn=100;pen_on=0}
    if(method == "WGAN"){L=1;hidden_size=40;zn=3;pen_on=0}
    if(method == "QR_m"){pen_on=1}
    if(method == "QR_nopen_m"){pen_on=0}
    
    source("./R code/model_fit.R")
    
    ymat_pen = ymat
    if(pen_on == 1){
        #n_test = nrow(Xt)
        n_test = 100
        #L0 = U0 = rep(0,n_test)
        #L1 = U1 = rep(0,n_test)
        Q_hat = matrix(0, dim(ymat_pen)[1], n_test)
        freq95 = rep(0,dim(ymat_pen)[1])
        U_hat = rep(0,dim(ymat_pen)[1])
        var_G = matrix(0,dim(ymat_pen)[1], n_test)
        no_cov = list()
        i = 1
        
        for(j in 1:dim(ymat_pen)[1]){
            no_cov[j] = c(no_cov[j],i)
        }
        
        for(j in 1:dim(ymat_pen)[1]){
            for(i in 1:n_test){
                Q_hat[j,i] = length(which(ymat_pen[j,,i] < yt[i]))/dim(ymat_pen)[2]
                l0 = quantile(ymat_pen[j,,i],0.025)
                u0 = quantile(ymat_pen[j,,i],0.975)
                if(yt[i] > l0 & yt[i] < u0){
                    freq95[j] = freq95[j] + 1/n_test
                }else{
                    no_cov[[j]] = c(no_cov[[j]],i)
                }
                var_G[j,i] = var(ymat_pen[j,,i])
            }
            q = mean(abs(sort(Q_hat[j,]) - (1:n_test)/n_test))
            U_hat[j] = q
        }
        
        ind_opt = which.min(discr)
        ind_opt = which.min(abs(freq95 - 0.95))
        ind_opt = which.min(U_hat)
        lam_opt = lam_cand0[ind_opt]
        
        #path0 = paste("/result/",n,"/",sep="")
        #name0 = paste(getwd(), path0, data_type, "_opt_lambda.png", sep="")
        #png(name0, width=1200, height=500)
    }
    if(pen_on == 1){ymat_pen_opt = ymat_pen[ind_opt,,]}
    if(pen_on == 0){ind_opt = 1;ymat_pen_opt = ymat_pen[ind_opt,,]}
    
    res[kk,,] = ymat_pen_opt
}

#### save the result ####
if(pen_on == 0){data_type = paste(model_type,"_",method,sep="")}
if(pen_on == 1){data_type = paste(model_type,"_",method,"_pen",sep="")}

#### Get the results ####
if(method == "QR_m"){
    data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K", "_COMPARISON_", "a_",fac, sep="")
}else{
    data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K", "_COMPARISON", sep="")
}

### Save path ####
path0 = paste("/result/",n,"/",sep="")
save_path = paste(getwd(), path0, data_type, "_gr.RData", sep="")
saveRDS(res, save_path)

