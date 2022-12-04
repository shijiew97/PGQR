#### Setting the working directory ####
#rm(list=ls())
#setwd("~/Dropbox/Shijie/GR")

#### Set the graph ####
name0 = paste(getwd(), "/result/real/", "RealData_NON-CROSS_QUANTILE.png", sep="")
num_plot = 4
if(num_plot == 4){
    png(name0, width=850, height=500)
    par(mfrow=c(2,2), mai=c(0.8,0.8,0.6,0.7))
}

library(randomcoloR)
library(colorspace)
library(quantreg)
n = 9
palette = distinctColorPalette(n)

#### Hyper-parameter to control the smoothness of nonparamteric QR ####
lam = 5.0
lam1 = 0.5

#### Plot the crossing quantiles ####
for(k in 1:2){
    
    method = "QR_m"
    if(k == 1){model_type = "motorcycles"}
    if(k == 2){model_type = "bmd"}
    
    if(model_type == "motorcycles"){
        
        library("adlift")
        data(motorcycledata)
        dat = motorcycledata
        #dat = data.frame(read.csv("./r code/fish.csv", sep=";"))
        #colnames(dat) = c("CIC0","SM1_Dz(Z)","GATS1i","NdsCH","NdssC","MLOGP","LC50")
        
        n = nrow(dat)
        #n = round(0.9*nrow(dat))
        ntest = nrow(dat) - n
        p = ncol(dat) -1
        
        set.seed(1567343)
        train_ind = base::sample(1:nrow(dat),size=n)
        dat_train = dat#[train_ind,]
        dat_test = dat#[-train_ind,]
        
        X = as.matrix(dat_train[,1:p]);y = as.matrix(dat_train[,p+1])
        Xt = as.matrix(dat_test[,1:p]);yt = as.matrix(dat_test[,p+1])
    }
    if(model_type == "bmd"){
        
        dat = read.table("./r code/BDM.txt",header=T)[,c("age", "spnbmd")]
        n = nrow(dat)
        
        for(i in 1:ncol(dat)){
            dat[,i]=as.numeric(dat[,i])
            dat[,i]=(dat[,i]-mean(dat[,i]))/sd(dat[,i])}
        
        #n = round(0.9*nrow(dat))
        ntest = nrow(dat) - n
        p = ncol(dat) -1
        
        set.seed(1567343)
        train_ind = base::sample(1:nrow(dat),size=n)
        dat_train = dat#[train_ind,]
        dat_test = dat#[-train_ind,]
        
        X = as.matrix(dat_train[,1:p]);y = as.matrix(dat_train[,p+1])
        Xt = as.matrix(dat_test[,1:p]);yt = as.matrix(dat_test[,p+1])
    }
    
    if(method == "QR"|method == "QR_m"){
        pen_on = 0
        if(method == "QR"){pen_on = 1}
        if(method == "QR_m"){pen_on = 1}
        if(method == "fGAN"){pen_on = 0}
        if(method == "WGAN"){pen_on = 0}
        
        if(pen_on == 0){data_type = paste(model_type,"_",method,sep="")}
        if(pen_on == 1){data_type = paste(model_type,"_",method,"_pen",sep="")}
        #data_type = paste(data_type,"_",p,"_", n sep="")
        #data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K", sep="")
        
        path0 = paste("/result/real/",sep="")
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
            
            
        }
        if(pen_on == 1){ymat_pen_opt = ymat_pen[ind_opt,,]}
        if(pen_on == 0){ind_opt = 1;ymat_pen_opt = ymat_pen[ind_opt,,]}
        
        ymat_QR = ymat_pen_opt
    }
    mul_quantile = discr[ind_opt,,]
    
    plot(X, y, xlab="", ylab="", cex.axis=1.2, col="grey")
    if(model_type == "motorcycles"){
        for(i in 1:9){
            quantile = seq(0.1,0.9,0.1)
            cex_c = c(1,rep(0.7,7),1)
            #lines(X, mul_quantile[i,], col=palette[i], lwd=1)
            #text(x=max(X)+0.5, y=mul_quantile[i,which.max(X)], labels=quantile[i], cex=cex_c[i])
            #grid()
            X = as.vector(X)
            res = rqss(y ~ qss(X, lambda=lam), tau=quantile[i])
            yhat = predict.rqss(res, newdata=data.frame(y,X))
            lines(X, yhat, col=darken(palette[i],0.5), lwd=1.2, lty=i)
        }
        title(xlab = "Time",line=2.5,cex.lab=1.3,family="mono")
        title(ylab = "Acceleration",line=2.5, cex.lab=1.3,family="mono")
        title(main = "Motorcycles Dataset",line=1.0,cex.main=1.3,family="mono")
        title(main = "(1) Quantile Regression",line=2.5,cex.main=1.7,family="mono")
    }
    if(model_type == "bmd"){
        for(i in 1:9){
            quantile = seq(0.1,0.9,0.1)
            cex_c = c(1,rep(0.7,7),1)
            sort_ind = sort(X, index.return=T)$ix
            #lines(sort(X), mul_quantile[i,sort_ind], col=palette[i], lwd=2)
            #text(x=max(X)+0.5, y=mul_quantile[i,which.max(X)], labels=quantile[i], cex=cex_c[i])
            #grid()
            X = as.vector(X)
            res = rqss(y ~ qss(X, lambda=lam1), tau=quantile[i])
            yhat = predict.rqss(res, newdata=data.frame(y,X))
            lines(sort(X), yhat[sort_ind], col=darken(palette[i],0.5), lwd=1.2, lty=i)
        }
        title(xlab = "Age (standardized)", line=2.5, cex.lab=1.3,family="mono")
        title(ylab = "BMD (standardized)", line=2.5, cex.lab=1.2,family="mono")
        title(main = "BMD Dataset", line=1.0, cex.main=1.3,family="mono")
    }
    
    plot(X, y, xlab="", ylab="", cex.axis=1.2, col="grey")
    if(model_type == "motorcycles"){
        for(i in 1:9){
            quantile = seq(0.1,0.9,0.1)
            cex_c = c(1,rep(0.7,7),1)
            lines(X, mul_quantile[i,], col=darken(palette[i],0.5), lwd=1.2, lty=i)
            #text(x=max(X)+0.5, y=mul_quantile[i,which.max(X)], labels=quantile[i], cex=cex_c[i])
            #grid()
            #abline(rq(y ~ X, tau=mul_quantile[i,], col=palette[i]))
        }
        title(xlab = "Time",line=2.5,cex.lab=1.3,family="mono")
        title(ylab = "Acceleration",line=2.5, cex.lab=1.3,family="mono")
        title(main = "Motorcycles Dataset",line=1.0,cex.main=1.3,family="mono")
        title(main = "(2) PGQR",line=2.5,cex.main=1.7,family="mono")
    }
    if(model_type == "bmd"){
        for(i in 1:9){
            quantile = seq(0.1,0.9,0.1)
            cex_c = c(1,rep(0.7,7),1)
            sort_ind = sort(X, index.return=T)$ix
            lines(sort(X), mul_quantile[i,sort_ind], col=darken(palette[i],0.5), lwd=1.2, lty=i)
            #text(x=max(X)+0.5, y=mul_quantile[i,which.max(X)], labels=quantile[i], cex=cex_c[i])
            #grid()
        }
        title(xlab = "Age (standardized)", line=2.5, cex.lab=1.3,family="mono")
        title(ylab = "BMD (standardized)", line=2.5, cex.lab=1.2,family="mono")
        title(main = "BMD Dataset", line=1.0, cex.main=1.3,family="mono")
    }
    legend("topright", inset=c(-0.15,0), legend=paste(seq(0.1,0.9,0.1),"",sep=""), 
           col=palette, lty=seq(1,9), bty="n", cex=0.9, xpd=TRUE, lwd=rep(1,9),
           pt.cex=1)   
}
dev.off()