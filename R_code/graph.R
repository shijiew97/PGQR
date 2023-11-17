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

#### Parameter Setting ####
N = 1
m = 1
test = 1
gpu_ind = 0
verb = 1
tsne_fit = 0
l1pen = 0

boot_size = 2000               #number of samples from PGQR
Seed = 12878                   #seed for simulation setting

lam_min = -30.0                #minimum of lambda candidate
lam_max = 0.1                  #maximum of lambda candidate

n = 2000                       #sample size
n0 = n                          
zn = 100
ntest = 100
p = 5.0                        #covariate dimension
if(model_type == "reg_P111"){p = 1.0}
if(model_type == "reg_linear"){p = 1.0}
S = n
fac = 1.0                      #alpha value


L = 2                          #number of hidden layers
batchnorm_on = 1
num_it = 10000*1               #number of iterations    
hidden_size = 1000             #number of hidden neurons
lr = 0.0001                    #learning rate
lr_power = 0.2                 #decaying learning rate
lrdecay = 1 
k = 100         
NN_type = "MLP"           
sigma0 = 1.0    



#### Get the data generation ####
source("./R_code/data_gen.R")

#### Fit the PGQR model ####
if(method == "fGAN"){L=2;hidden_size=40;zn=3;pen_on=0}
if(method == "fGAN_C"){L=3;hidden_size=1000;zn=100;pen_on=0}
if(method == "WGAN"){L=1;hidden_size=40;zn=3;pen_on=0}
if(method == "QR_m"){pen_on=1}
if(method == "QR_nopen_m"){pen_on=0}

#source("./R_code/model_fit.R")

#### whether to include following model in plot ####
flex = 0                       #flexcode method
fGAN = 1                       #GCDS 
fGAN_C = 1                     #deep-GCDS
WGAN = 1                       #WGCS

#### Get the results ####
if(flex == 1){
    flex_fit = flex_eval(X,y,Xt,yt);flex_cde = flex_fit[[2]];flex_grid = flex_fit[[1]]}

num_it0 = 10000*5
if(fGAN == 1){
    method = "fGAN"
    if(method == "fGAN"){
        pen_on = 0
        if(method == "QR"){pen_on = 1}
        if(method == "fGAN"){pen_on = 0}
        if(method == "WGAN"){pen_on = 0}
        
        if(model_type == "reg_multimode"){num_it0=10000*2}
        if(pen_on == 0){data_type = paste(model_type,"_",method,sep="")}
        if(pen_on == 1){data_type = paste(model_type,"_",method,"_pen",sep="")}
        #data_type = paste(data_type,"_",p,"_", n sep="")
        data_type = paste(data_type,"_",p,"_", n, "_", num_it0/1000, "K", sep="")
        
        path0 = paste("/result/",n,"/",sep="")
        read_path = paste(getwd(), path0, data_type, "_gr.RData", sep="")
        gr_res = readRDS(read_path)
        ymat_pen = gr_res[[5]]
        
        X = gr_res[[1]];y = gr_res[[2]];Xt = gr_res[[3]];yt = gr_res[[4]]
        discr = gr_res[[6]];lam_cand0 = gr_res[[7]];sigma0 = gr_res[[8]]
        p = gr_res[[9]]
        
        if(pen_on == 1){
            n_test = nrow(Xt)
            
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
            
            #        path0 = paste("/result/",n,"/",sep="")
            #        name0 = paste(getwd(), path0, data_type, "_opt_lambda.png", sep="")
            #        png(name0, width=1200, height=500)
            
            
            par(mfrow=c(1,2))
            
            c_sd = apply(ymat_pen, c(1,3), sd)
            csd = data.frame("con_std"=as.vector(unlist(c_sd, use.names=F)),
                             "label"=rep(round(lam_cand0,1), dim(c_sd)[2]))
            boxplot(con_std~label, data=csd, col=rgb(0.5,0.5,1),
                    xlab="lambda", ylab="conditional std")
            abline(v=ind_opt, col="red", lty=2, lwd=1)
            #abline(v=50, col="blue", lty=2, lwd=1)
            
            title = paste(data_type, ": optimial lambdal <<", round(lam_opt,2), ">>",
                          sep="")
            plot(x=lam_cand0 , y=U_hat, pch=4, xlab="lambda candidate",
                 ylab="CR-stat", main=title, cex.main=0.9)
            abline(v=lam_opt, col="red", lty=2)
            points(x=lam_opt, y=U_hat[ind_opt], col="red", pch=4, lwd=2)
            
            #        dev.off()
        }
        if(pen_on == 1){ymat_pen_opt = ymat_pen[ind_opt,,]}
        if(pen_on == 0){ind_opt = 1;ymat_pen_opt = ymat_pen[ind_opt,,]}
        
        ymat_fGAN = ymat_pen_opt
    }
}

num_it00 = 10000*5
if(fGAN_C == 1){
    method = "fGAN_C"
    if(method == "fGAN_C"){
        pen_on = 0
        if(method == "QR"){pen_on = 1}
        if(method == "fGAN"){pen_on = 0}
        if(method == "WGAN"){pen_on = 0}
        if(method == "fGAN_C"){pen_on = 0}
        
        if(pen_on == 0){data_type = paste(model_type,"_",method,sep="")}
        if(pen_on == 1){data_type = paste(model_type,"_",method,"_pen",sep="")}
        #data_type = paste(data_type,"_",p,"_", n sep="")
        data_type = paste(data_type,"_",p,"_", n, "_", num_it00/1000, "K", sep="")
        
        path0 = paste("/result/",n,"/",sep="")
        read_path = paste(getwd(), path0, data_type, "_gr.RData", sep="")
        gr_res = readRDS(read_path)
        ymat_pen = gr_res[[5]]
        
        X = gr_res[[1]];y = gr_res[[2]];Xt = gr_res[[3]];yt = gr_res[[4]]
        discr = gr_res[[6]];lam_cand0 = gr_res[[7]];sigma0 = gr_res[[8]]
        p = gr_res[[9]]
        
        if(pen_on == 1){
            n_test = nrow(Xt)
            
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
            
            #        path0 = paste("/result/",n,"/",sep="")
            #        name0 = paste(getwd(), path0, data_type, "_opt_lambda.png", sep="")
            #        png(name0, width=1200, height=500)
            
            
            par(mfrow=c(1,2))
            
            c_sd = apply(ymat_pen, c(1,3), sd)
            csd = data.frame("con_std"=as.vector(unlist(c_sd, use.names=F)),
                             "label"=rep(round(lam_cand0,1), dim(c_sd)[2]))
            boxplot(con_std~label, data=csd, col=rgb(0.5,0.5,1),
                    xlab="lambda", ylab="conditional std")
            abline(v=ind_opt, col="red", lty=2, lwd=1)
            #abline(v=50, col="blue", lty=2, lwd=1)
            
            title = paste(data_type, ": optimial lambdal <<", round(lam_opt,2), ">>",
                          sep="")
            plot(x=lam_cand0 , y=U_hat, pch=4, xlab="lambda candidate",
                 ylab="CR-stat", main=title, cex.main=0.9)
            abline(v=lam_opt, col="red", lty=2)
            points(x=lam_opt, y=U_hat[ind_opt], col="red", pch=4, lwd=2)
            
            #        dev.off()
        }
        if(pen_on == 1){ymat_pen_opt = ymat_pen[ind_opt,,]}
        if(pen_on == 0){ind_opt = 1;ymat_pen_opt = ymat_pen[ind_opt,,]}
        
        ymat_fGAN_C = ymat_pen_opt
    }
}

num_it000 = 10000*5
if(WGAN == 1){
    method = "WGAN"
    graph = 1
    pen_on = 0
    if(method == "QR"){pen_on = 1}
    if(method == "fGAN"){pen_on = 0}
    if(method == "WGAN"){pen_on = 0}
    #pen_on = 0
    
    if(pen_on == 0){data_type = paste(model_type,"_",method,sep="")}
    if(pen_on == 1){data_type = paste(model_type,"_",method,"_pen",sep="")}
    #data_type = paste(data_type,"_",p,"_", n sep="")
    data_type = paste(data_type,"_",p,"_", n, "_", num_it000/1000, "K", sep="")
    
    path0 = paste("/result/",n,"/",sep="")
    read_path = paste(getwd(), path0, data_type, "_gr.RData", sep="")
    gr_res = readRDS(read_path)
    ymat_pen = gr_res[[5]]
    
    X = gr_res[[1]];y = gr_res[[2]];Xt = gr_res[[3]];yt = gr_res[[4]]
    discr = gr_res[[6]];lam_cand0 = gr_res[[7]];sigma0 = gr_res[[8]]
    p = gr_res[[9]]
    
    if(pen_on == 1){
        n_test = nrow(Xt)
        
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
        
        if(graph == 1){
            path0 = paste("/result/",n,"/",sep="")
            name0 = paste(getwd(), path0, data_type, "_opt_lambda.png", sep="")
            png(name0, width=850, height=250)
            
            
            par(mfrow=c(1,4))
            
            c_sd = apply(ymat_pen, c(1,3), sd)
            csd = data.frame("con_std"=as.vector(unlist(c_sd, use.names=F)),
                             "label"=rep(round(lam_cand0,1), dim(c_sd)[2]))
            if(model_type == "reg_linear"){
                boxplot(con_std~label, data=csd, col=rgb(0.5,0.5,1),
                        xlab="log-lambda", ylab="conditional std",cex.lab=1.3, ylim=c(0,0.5))
                abline(h=0.1, col="orange", lty=2)
            }else{
                boxplot(con_std~label, data=csd, col=rgb(0.5,0.5,1),
                        xlab="log-lambda", ylab="conditional std",cex.lab=1.3, ylim=c(0,5))}
            abline(v=ind_opt, col="red", lty=2, lwd=1)
            
            plot(x=lam_cand0,y=freq95,xlab="log-lambda", ylab="Covergae rate",cex.lab=1.3,pch=16)
            abline(v=lam_opt, col="red", lty=2, lwd=1)
            #abline(h=0.1, col="orange", lty=2)
            #abline(v=ind_opt, col="red", lty=2, lwd=1)
            
            title = paste("optimial log-lambdal <<", round(lam_opt,2), ">>",
                          sep="")
            plot(x=lam_cand0 , y=U_hat, pch=4, xlab="log-lambda candidate",
                 ylab="CR-stat", main=title, cex.main=1.3,cex.lab=1.3)
            abline(v=lam_opt, col="red", lty=2)
            points(x=lam_opt, y=U_hat[ind_opt], col="red", pch=4, lwd=2)
            
            hist(Q_hat[ind_opt,],xlab="",freq=F,col=rgb(1,0.6,0,0.5),
                 main="Distribution of pi",cex.main=1.3)
            
            dev.off()
        }
    }
    if(pen_on == 1){ymat_pen_opt = ymat_pen[ind_opt,,]}
    if(pen_on == 0){ind_opt = 1;ymat_pen_opt = ymat_pen[ind_opt,,]}
    
    ymat_WGAN = ymat_pen_opt
}

##Get the monotone
method = "QR_m"
if(method == "QR_m"){
    graph = 1
    pen_on = 0
    if(method == "QR"){pen_on = 1}
    if(method == "QR_m"){pen_on = 1}
    if(method == "fGAN"){pen_on = 0}
    if(method == "WGAN"){pen_on = 0}
    #pen_on = 0
    #num_it = 10000
    
    if(pen_on == 0){data_type = paste(model_type,"_",method,sep="")}
    if(pen_on == 1){data_type = paste(model_type,"_",method,"_pen",sep="")}
    
    #data_type = paste(data_type,"_",p,"_", n sep="")
    num_it = 10000
    sensitive = 0
    if(sensitive==0){data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K", sep="")}
    if(sensitive==1){data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K","_a_",fac,sep="")}
    
    
    path0 = paste("/result/",n,"/",sep="")
    read_path = paste(getwd(), path0, data_type, "_gr.RData", sep="")
    gr_res = readRDS(read_path)
    ymat_pen = gr_res[[5]]
    
    X = gr_res[[1]];y = gr_res[[2]];Xt = gr_res[[3]];yt = gr_res[[4]]
    discr = gr_res[[6]];lam_cand0 = gr_res[[7]];sigma0 = gr_res[[8]]
    p = gr_res[[9]]
    
    if(pen_on == 1){
        n_test = nrow(Xt)
        
        #L0 = U0 = rep(0,n_test)
        #L1 = U1 = rep(0,n_test)
        Q_hat = matrix(0, dim(ymat_pen)[1], n_test)
        freq95 = rep(0,dim(ymat_pen)[1])
        U_hat = rep(0,dim(ymat_pen)[1])
        ks_hat = rep(0,dim(ymat_pen)[1])
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
            #q = mean(abs(sort(Q_hat[j,]) - (1:n_test)/n_test))
            q = mean((((1:n_test)-1/2)/n_test-sort(Q_hat[j,]))^2)
            ks_hat[j] = stats::ks.test(x=Q_hat[j,], y=punif)$statistic
            U_hat[j] = q
        }
        
        ind_opt = which.min(discr)
        ind_opt = which.min(abs(freq95 - 0.95))
        ind_opt = which.min(U_hat)
        #ind_opt = which.min(ks_hat)
        lam_opt = lam_cand0[ind_opt]
        
        if(graph == 1){
            path0 = paste("/result/",n,"/",sep="")
            name0 = paste(getwd(), path0, data_type, "_opt_lambda.png", sep="")
            png(name0, width=850, height=250)
            
            
            par(mfrow=c(1,4))
            
            c_sd = apply(ymat_pen, c(1,3), sd)
            csd = data.frame("con_std"=as.vector(unlist(c_sd, use.names=F)),
                             "label"=rep(round(lam_cand0,1), dim(c_sd)[2]))
            if(model_type == "reg_linear"){
                boxplot(con_std~label, data=csd, col=rgb(0.5,0.5,1),
                        xlab="log-lambda", ylab="conditional std",cex.lab=1.3, ylim=c(0,0.5))
                abline(h=0.1, col="orange", lty=2)
            }else{
                boxplot(con_std~label, data=csd, col=rgb(0.5,0.5,1),
                        xlab="log-lambda", ylab="conditional std",cex.lab=1.3)}
            abline(v=ind_opt, col="red", lty=2, lwd=1)
            
            plot(x=lam_cand0,y=freq95,xlab="log-lambda", ylab="Covergae rate",cex.lab=1.3,pch=16)
            abline(v=lam_opt, col="red", lty=2, lwd=1)
            #abline(h=0.1, col="orange", lty=2)
            #abline(v=ind_opt, col="red", lty=2, lwd=1)
            
            title = paste("optimial log-lambdal <<", round(lam_opt,2), ">>",
                          sep="")
            plot(x=lam_cand0 , y=U_hat, pch=4, xlab="log-lambda candidate",
                 ylab="CR-stat", main=title, cex.main=1.3,cex.lab=1.3)
            abline(v=lam_opt, col="red", lty=2)
            points(x=lam_opt, y=U_hat[ind_opt], col="red", pch=4, lwd=2)
            
            hist(Q_hat[ind_opt,],xlab="",freq=F,col=rgb(1,0.6,0,0.5),
                 main="Distribution of pi",cex.main=1.3)
            
            dev.off()
        }
    }
    #ind_opt = 20
    if(pen_on == 1){ymat_pen_opt = ymat_pen[ind_opt,,]}
    if(pen_on == 0){ind_opt = 1;ymat_pen_opt = ymat_pen[ind_opt,,]}
    
    ymat_QR = ymat_pen_opt
}


if(method == "QR_nopen_m"){
    pen_on = 0
    if(method == "QR"){pen_on = 1}
    if(method == "QR_nopen"){pen_on = 0}
    if(method == "fGAN"){pen_on = 0}
    if(method == "WGAN"){pen_on = 0}
    #pen_on = 0
    
    if(pen_on == 0){data_type = paste(model_type,"_",method,sep="")}
    if(pen_on == 1){data_type = paste(model_type,"_",method,"_pen",sep="")}
    #data_type = paste(data_type,"_",p,"_", n sep="")
    data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K", sep="")
    
    path0 = paste("/result/",n,"/",sep="")
    read_path = paste(getwd(), path0, data_type, "_gr.RData", sep="")
    gr_res = readRDS(read_path)
    ymat_pen = gr_res[[5]]
    
    X = gr_res[[1]];y = gr_res[[2]];Xt = gr_res[[3]];yt = gr_res[[4]]
    discr = gr_res[[6]];lam_cand0 = gr_res[[7]];sigma0 = gr_res[[8]]
    p = gr_res[[9]]
    
    if(pen_on == 1){
        n_test = nrow(Xt)
        
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
        
        path0 = paste("/result/",n,"/",sep="")
        name0 = paste(getwd(), path0, data_type, "_opt_lambda.png", sep="")
        png(name0, width=850, height=250)
        
        
        par(mfrow=c(1,3))
        
        c_sd = apply(ymat_pen, c(1,3), sd)
        csd = data.frame("con_std"=as.vector(unlist(c_sd, use.names=F)),
                         "label"=rep(round(lam_cand0,1), dim(c_sd)[2]))
        #boxplot(con_std~label, data=csd, col=rgb(0.5,0.5,1),
        #        xlab="log-lambda", ylab="conditional std",cex.lab=1.3,ylim=c(0, 0.5))
        plot(x=lam_cand0,y=freq95,xlab="log-lambda", ylab="Covergae rate",cex.lab=1.3,pch=16)
        abline(v=lam_opt, col="red", lty=2, lwd=1)
        #abline(h=0.1, col="orange", lty=2)
        #abline(v=50, col="blue", lty=2, lwd=1)
        
        title = paste("optimial log-lambdal <<", round(lam_opt,2), ">>",
                      sep="")
        plot(x=lam_cand0 , y=U_hat, pch=4, xlab="log-lambda candidate",
             ylab="CR-stat", main=title, cex.main=1.3,cex.lab=1.3)
        abline(v=lam_opt, col="red", lty=2)
        points(x=lam_opt, y=U_hat[ind_opt], col="red", pch=4, lwd=2)
        
        hist(Q_hat[ind_opt,],xlab="",freq=F,col=rgb(1,0.6,0,0.5),
             main="Distribution of pi",cex.main=1.3)
        
        dev.off()
    }
    if(pen_on == 1){ymat_pen_opt = ymat_pen[ind_opt,,]}
    if(pen_on == 0){ind_opt = 1;ymat_pen_opt = ymat_pen[ind_opt,,]}
    
    ymat_QR = ymat_pen_opt
}


#### Get the plot ####
n0 = dim(ymat_pen_opt)[1]
num_plot = 3
if(model_type = "reg_norm"){num_plot = 6}

#### Plot setting ####
path0 = paste("/result/",n,"/",sep="")
name0 = paste(getwd(), path0, data_type, "_dist_compareL.png", sep="")
if(num_plot == 9){
    png(name0, width=850, height=500)
    par(mfrow=c(3,3), mai=c(0.4,0.4,0.2,0.2))
}
if(num_plot == 6){
    png(name0, width=1450, height=614)
    par(mfrow=c(2,3), mai=c(0.4,0.4,0.2,0.2))
}
if(num_plot == 3){
    png(name0, width=1450, height=400)
    par(mfrow=c(1,3), mai=c(0.4,0.4,0.2,0.2))
}
if(num_plot == 4){
    png(name0, width=900, height=250)
    par(mfrow=c(2,2), mai=c(0.4,0.4,0.2,0.2))
}

#### Graph results ####
if(num_plot != 4){
    for(i in (num_plot+1):(num_plot+num_plot)){
        
        if(model_type != "reg_skew"){j = 2*i^2 + 1}
        else{j = 2*i^2 + 2}
        if(model_type == "reg_norm"){j = i^2 + 2*i}
        if(model_type=="reg_P111"){
            jj0 = quantile(1:length(Xt), probs=c(0.05,0.10,0.25,0.5,0.75,0.9))
            j = jj0[i]
        }
        yt_QR = ymat_QR[,j]
        
        #remove the extreme quantiles at tails.
        if(method == "QR"|method == "QR_m"){
            quant = quantile(yt_QR, probs=c(0.02,0.98))
            if(model_type == "reg_skewV2"){quant = quantile(yt_QR, probs=c(0.004,0.996))}
            if(model_type == "reg_norm"){quant = quantile(yt_QR, probs=c(0.004,0.996))}
            ind_out = c(which(yt_QR>=quant[2]), which(yt_QR<=quant[1]))
            yt_QR = yt_QR[-ind_out]
            n0 = length(yt_QR)
        }
        
        X_gen = matrix(rep(Xt[j,], each=n0), n0, p)
        Z_gen = z_true(X_gen,n0)
        y_gen = gen_true(X_gen, Z_gen)
        
        c0 = c(0.5,0.5,0.4,0.5,0.4,0.4,0.4,0.5,0.4)
        if(model_type=="reg_multimode"){c0 = c(0.6,0.6,0.5,0.3,0.2,0.2,0.5,0.5,0.4)}#multi-modal
        if(model_type=="reg_skew"){c0 = c(0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.4)}#skew
        if(model_type=="reg_skewV2"){c0 = c(0.4,0.5,0.6,0.6,0.6,0.4,0.4,0.4,0.4)}#skewV2
        if(model_type=="reg_P111"){c0 = c(0.3,0.3,0.3,0.4,0.3,0.3,0.3,0.2,0.2)}
        if(model_type=="reg_linear"){c0 = c(0.5,0.5,0.4,0.5,0.4,0.4,0.4,0.5,0.4)}
        if(model_type=="reg_nonparam"){c0 = rep(0.6,10)}
        if(model_type=="reg_simple"){c0 = c(0.6,0.6,0.6,0.6,0.6,0.6,0.4,0.5,0.4)}
        if(p == 50){c0 = c(0.5,0.5,0.6,0.6,0.6,0.6,0.4,0.5,0.4)}
        if(model_type=="reg_norm"){c0 = rep(0.5, 100)}
            
        #c0 = rep(0.3,10)
        den_t = density(y_gen, bw=0.3*sd(y_gen))
        den_QR = density(yt_QR, bw=c0[i]*sd(yt_QR))
        #den_QR = density(yt_QR, bw=0.2*sd(yt_QR))
        
        if(fGAN == 1){
            yt_fGAN = ymat_fGAN[,j]
            if(model_type=="reg_P111"){c0 = rep(0.2,10)}
            den_fGAN = density(yt_fGAN, bw=0.4*sd(yt_fGAN))}
        if(fGAN_C == 1){
            yt_fGAN_C = ymat_fGAN_C[,j]
            den_fGAN_C = density(yt_fGAN_C, bw=c0[i]*sd(yt_fGAN_C))}
        if(WGAN == 1){
            yt_WGAN = ymat_WGAN[,j]
            den_WGAN = density(yt_WGAN, bw=0.4*sd(yt_WGAN))}
        center = yt[j]#gen_true(matrix(Xt[j,], ncol=p), 0)
        
        
        
        if(model_type=="reg_skew"){
            Ylim = c(0, 0.4)
            if(i == 4){Xlim=c(-10,20)}
            if(i == 5){Xlim=c(-20,10)}
            if(i == 6){Xlim=c(-20,25)}
        }
        if(model_type=="reg_multimode"){
            Ylim = c(0, 0.4)
            if(i == 4){Xlim=c(-5,10)}
            if(i == 5){Xlim=c(-15,5)}
            if(i == 6){Xlim=c(-5,20)}
        }
        if(model_type=="reg_P111"){
            Xlim = c(-30, 30)
            Ylim = c(0, 0.12)
            if(i== 4){
                Xlim = c(-5, 5)
                Ylim = c(0, 4)}
            if(i == 6){Xlim=c(-60,75)}
        }
        if(model_type=="reg_nonparam"){
            Ylim = c(0, 0.6)
            Xlim = c(-5, 10)
        }
        if(model_type=="reg_linear"){
            Ylim = c(0, 5)
            Xlim = c(-3, 0)
        }
        if(model_type=="reg_norm"){
            Ylim = c(0, 0.2)
            Xlim = c(-40, 40)
        }
        
        
        sd1 = round(sd(y_gen),2);c1 = round(mean(y_gen),1)
        sd3 = round(sd(yt_QR),2);c2 = round(mean(yt_QR),1)
        
        if(flex == 2){
            main1 = paste("true_sd:",
                          sd1, ", est_sd:",sd3,
                          ", flex_sd", round(flex_fit[[3]][2],2),
                          sep="")
        }
        
        plot(den_t, col="black", xlim=Xlim, ylim=Ylim, xlab="", ylab="", main="",lwd=3, cex.axis=2.0)
        lines(den_QR, col=rgb(0,0,1,1), lwd=3, lty=1)
        points(den_QR, col=rgb(0,0,1,1), pch=c(16, rep(NA, 80)), cex=3.5, lwd=2)
        if(fGAN == 1){
            lines(den_fGAN, col=rgb(1,0,0,0.9), lwd=2, lty=3)
            points(den_fGAN, col=rgb(1,0,0,0.9), pch=c(2, rep(NA, 80)), cex=3.5, lwd=2)
        }
        if(fGAN_C == 1){
            lines(den_fGAN_C, col=rgb(0.5,0,0.5,0.9), lwd=2, lty=4)
            points(den_fGAN_C, col=rgb(0.5,0,0.5,0.9), pch=c(3, rep(NA, 80)), cex=3.5, lwd=2)}
        #if(WGAN == 1){lines(den_WGAN, col=rgb(0.5,0,0.5,0.9), lwd=2, lty=2)}
        if(flex == 1){lines(flex_grid, flex_cde[j,], col=rgb(1,0.6,0,0.9),lwd=2,lty=2)}
        #lines(NNK_grid, NNK_cde[j,], col=rgb(1,0.6,0,0.4),lwd=2,lty=1)
        #lines(den2, col=rgb(1,0,0,0.6), lwd=2, lty=2)
        points(x=center, y=0, col="black", pch=4, lwd=2, cex=2)
        #if(i !=num_plot){text(x=19,y=Ylim[2]-0.05,labels=j,cex=1.2)}
        if(model_type=="reg_norm"){
            text(-40, 0.18, expression("||X||"[1]), pos=4, cex=3)
            text(-30, 0.18, paste("=", round(sum(abs(Xt[j,])), 1)), pos=4, cex=3)
        }
        if(WGAN == 1){
            lines(den_WGAN, col=3, lwd=2, lty=5)
            points(den_WGAN, col=3, pch=c(4, rep(NA, 80)), cex=3.5, lwd=2)}
    }
    
    if(fGAN == 0){legend("topright", c("True","PGQR"),
                         col=c("black",rgb(0,0,1)),
                         lwd=c(2,2), lty=c(1,2), bty="n", cex=3.0,
                         pch=c(NA,16))}
    else{
        legend("topright", c("True","PGQR","GCDS","WGCS","deep-GCDS"),
               col=c("black",rgb(0,0,1),rgb(1,0,0),3,rgb(0.5,0,0.5)),
               lwd=c(2,2,2,2,2), lty=c(1,2,3,4,5), bty="n", cex=2.5,
               pch=c(NA,16,2,4,3))
    }
    dev.off()
}