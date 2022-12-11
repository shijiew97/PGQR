#### Setting the working directory ####
#rm(list=ls())
#setwd("~/Dropbox/Shijie/GR")

#### Get the APL results ####
method = "QR_m"
model_type = "APL"
if(method == "QR_m"){
    graph = 1
    pen_on = 0
    p = 1
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
    #sensitive = 0
    #if(sensitive==0){data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K", sep="")}
    #if(sensitive==1){data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K","_a_",fac,sep="")}
    
    
    path0 = "/result/real/"
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
            path0 = "/result/real/"
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
    #ind_opt = 20
    if(pen_on == 1){ymat_pen_opt = ymat_pen[ind_opt,,]}
    if(pen_on == 0){ind_opt = 1;ymat_pen_opt = ymat_pen[ind_opt,,]}
    
    ymat_QR = ymat_pen_opt
}

#### Get the plot ####
name0 = paste(getwd(), "/result/real/", "Real_Data_bimodal.png", sep="")
num_plot = 2
if(num_plot == 2){
    png(name0, width=1200, height=500)
    par(mfrow=c(1,2), mai=c(0.8,0.8,0.6,0.7))
}

#### Two test points at 45.3 and 77.5 ####
for(i in 1:num_plot){
    
    if(i == 2){j = 1330}
    if(i == 1){j = 76}
    
    #Get the PGQR results
    yt_QR = ymat_QR[,j]
    if(method == "QR"|method == "QR_m"){
        quant = quantile(yt_QR, probs=c(0.02,0.98))
        if(model_type == "reg_skewV2"){quant = quantile(yt_QR, probs=c(0.004,0.996))}
        ind_out = c(which(yt_QR>=quant[2]), which(yt_QR<=quant[1]))
        yt_QR = yt_QR[-ind_out]
        n0 = length(yt_QR)
    }
    
    #Plot the background
    plot(X,y, col=rgb(0,0,0,0.4), pch=19, ylab="", xlab="")
    title(xlab = "Weight",line=2.5,cex.lab=1.3,family="mono")
    title(ylab = "AP-L/F Ratio",line=2.5, cex.lab=1.3,family="mono")
    #title(main = "Motorcycles Dataset",line=1.0,cex.main=1.3,family="mono")
    
    bw = 0.25
    abline(v=Xt[j], col="red", lwd=2)
    d = density(yt_QR, bw=bw)
    lines(50*d$y+Xt[j], d$x, lwd=4, col=rgb(0,0.5,1,1))
    polygon(50*d$y+Xt[j], d$x, col=rgb(0,0.2,1,0.2))
}
dev.off()