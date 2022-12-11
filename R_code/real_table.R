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

#### whether to include following model####
flex = 0                       #flexcode method
fGAN = 1                       #GCDS 
fGAN_C = 1                     #deep-GCDS
WGAN = 1                       #WGCS

#### Real dataset ####
if(model_type == "noise"){
    dat = data.frame(read.table("./R_code/noise.dat", skip=3))
    colnames(dat) = c("freq","angle","chord_len","velocity","thickness","sound")
    
    for(i in 1:ncol(dat)){dat[,i]=as.numeric(dat[,i])}
    n = round(0.8*nrow(dat))
    ntest = nrow(dat) - n
    p = ncol(dat) -1
    
    for(i in 1:p){dat[,i]=(dat[,i]-mean(dat[,i]))/sd(dat[,i])}
    
    set.seed(153)
    train_ind = base::sample(1:nrow(dat),size=n)
    dat_train = dat[train_ind,]
    dat_test = dat[-train_ind,]
    
    X = as.matrix(dat_train[,1:p]);y = as.matrix(dat_train[,p+1])
    Xt = as.matrix(dat_test[,1:p]);yt = as.matrix(dat_test[,p+1])
    
    Seed = 98432}
if(model_type == "machine"){
    
    dat = data.frame(read.csv("./R_code/machine.csv",sep=";"))
    colnames(dat) = c("Load-Current","PF","Error","DIF","IF")
    
    
    for(i in 1:ncol(dat)){dat[,i]=gsub(",",".",dat[,i]);dat[,i]=as.numeric(dat[,i])}
    n = round(0.8*nrow(dat))
    ntest = nrow(dat) - n
    p = ncol(dat)
    
    set.seed(12345)
    train_ind = base::sample(1:nrow(dat),size=n)
    dat_train = dat[train_ind,]
    dat_test = dat[-train_ind,]
    
    X = as.matrix(dat_train[,2:p]);y = as.matrix(dat_train[,1])
    Xt = as.matrix(dat_test[,2:p]);yt = as.matrix(dat_test[,1])
    
    Seed = 486464}
if(model_type == "fish"){
    
    dat = data.frame(read.csv("./R_code/fish.csv", sep=";"))
    colnames(dat) = c("CIC0","SM1_Dz(Z)","GATS1i","NdsCH","NdssC","MLOGP","LC50")
    
    n = round(0.8*nrow(dat))
    ntest = nrow(dat) - n
    p = ncol(dat) -1
    
    set.seed(1567343)
    train_ind = base::sample(1:nrow(dat),size=n)
    dat_train = dat[train_ind,]
    dat_test = dat[-train_ind,]
    
    X = as.matrix(dat_train[,1:p]);y = as.matrix(dat_train[,p+1])
    Xt = as.matrix(dat_test[,1:p]);yt = as.matrix(dat_test[,p+1])
    
    Seed = 98432
}


#### Initialize table ####
set.seed(Seed)
n0 = round(nrow(Xt)*0.5)
val_ind = base::sample(1:nrow(Xt),size=n0)
CI_table = matrix(0, nrow=(nrow(Xt)-n0), ncol=4)

#### Read the results ####
if(flex == 1){
    flex_fit = flex_eval(X,y,Xt,yt);flex_cde = flex_fit[[2]];flex_grid = flex_fit[[1]];flex_thres=flex_fit[[4]]}
if(fGAN == 1){
    method = "fGAN"
    if(method == "fGAN"){
        pen_on = 0
        if(method == "QR"){pen_on = 1}
        if(method == "fGAN"){pen_on = 0}
        if(method == "WGAN"){pen_on = 0}
        
        if(pen_on == 0){data_type = paste(model_type,"_",method,sep="")}
        if(pen_on == 1){data_type = paste(model_type,"_",method,"_pen",sep="")}
        #data_type = paste(data_type,"_",p,"_", n sep="")
        #data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K", sep="")
        
        path0 = paste("/result/real/",sep="")
        read_path = paste(getwd(), path0, data_type, "_gr.RData", sep="")
        gr_res = readRDS(read_path)
        ymat_pen0 = gr_res[[5]]
        
        X = gr_res[[1]];y = gr_res[[2]];Xt0 = gr_res[[3]];yt0 = gr_res[[4]]
        discr = gr_res[[6]];lam_cand0 = gr_res[[7]];sigma0 = gr_res[[8]]
        p = gr_res[[9]]
        
        
        Xt = Xt0[val_ind,]
        Xtt = Xt0[-val_ind,]
        
        yt = yt0[val_ind,]
        ytt = yt0[-val_ind,]
        
        ymat_pen = ymat_pen0[,,val_ind]
        ymat_pen00 = ymat_pen0[,,-val_ind]
        
        
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
            
            #        dev.off()
        }
        if(pen_on == 1){ymat_pen_opt = ymat_pen00[ind_opt,,]}
        if(pen_on == 0){ind_opt = 1;ymat_pen_opt = ymat_pen00[ind_opt,,]}
        
        ymat_fGAN = ymat_pen_opt
    }
}
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
        #data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K", sep="")
        
        path0 = paste("/result/real/",sep="")
        read_path = paste(getwd(), path0, data_type, "_gr.RData", sep="")
        gr_res = readRDS(read_path)
        ymat_pen0 = gr_res[[5]]
        
        X = gr_res[[1]];y = gr_res[[2]];Xt0 = gr_res[[3]];yt0 = gr_res[[4]]
        discr = gr_res[[6]];lam_cand0 = gr_res[[7]];sigma0 = gr_res[[8]]
        p = gr_res[[9]]
        
        Xt = Xt0[val_ind,]
        Xtt = Xt0[-val_ind,]
        
        yt = yt0[val_ind,]
        ytt = yt0[-val_ind,]
        
        ymat_pen = ymat_pen0[,,val_ind]
        ymat_pen00 = ymat_pen0[,,-val_ind]
        
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
            
            path0 = paste("/result/real","/",sep="")
            name0 = paste(getwd(), path0, data_type, "_opt_lambda.png", sep="")
            png(name0, width=850, height=250)
            
            
        }
        if(pen_on == 1){ymat_pen_opt = ymat_pen00[ind_opt,,]}
        if(pen_on == 0){ind_opt = 1;ymat_pen_opt = ymat_pen00[ind_opt,,]}
        
        ymat_fGAN_C = ymat_pen_opt
    }
}
if(WGAN == 1){
    method = "WGAN"
    if(method == "WGAN"){
        pen_on = 0
        if(method == "QR"){pen_on = 1}
        if(method == "fGAN"){pen_on = 0}
        if(method == "WGAN"){pen_on = 0}
        
        if(pen_on == 0){data_type = paste(model_type,"_",method,sep="")}
        if(pen_on == 1){data_type = paste(model_type,"_",method,"_pen",sep="")}
        #data_type = paste(data_type,"_",p,"_", n sep="")
        #data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K", sep="")
        
        path0 = paste("/result/real/",sep="")
        read_path = paste(getwd(), path0, data_type, "_gr.RData", sep="")
        gr_res = readRDS(read_path)
        ymat_pen0 = gr_res[[5]]
        
        X = gr_res[[1]];y = gr_res[[2]];Xt0 = gr_res[[3]];yt0 = gr_res[[4]]
        discr = gr_res[[6]];lam_cand0 = gr_res[[7]];sigma0 = gr_res[[8]]
        p = gr_res[[9]]
        
        
        Xt = Xt0[val_ind,]
        Xtt = Xt0[-val_ind,]
        
        yt = yt0[val_ind,]
        ytt = yt0[-val_ind,]
        
        ymat_pen = ymat_pen0[,,val_ind]
        ymat_pen00 = ymat_pen0[,,-val_ind]
        
        
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
            
            #        dev.off()
        }
        if(pen_on == 1){ymat_pen_opt = ymat_pen00[ind_opt,,]}
        if(pen_on == 0){ind_opt = 1;ymat_pen_opt = ymat_pen00[ind_opt,,]}
        
        ymat_WGAN = ymat_pen_opt
    }
    
}

method = "QR_m"
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
    ymat_pen0 = gr_res[[5]]
    
    X = gr_res[[1]];y = gr_res[[2]];Xt0 = gr_res[[3]];yt0 = gr_res[[4]]
    discr = gr_res[[6]];lam_cand0 = gr_res[[7]];sigma0 = gr_res[[8]]
    p = gr_res[[9]]
    
    #get validation set
    Xt = Xt0[val_ind,]
    Xtt = Xt0[-val_ind,]
    
    yt = yt0[val_ind,]
    ytt = yt0[-val_ind,]
    
    ymat_pen = ymat_pen0[,,val_ind]
    ymat_pen00 = ymat_pen0[,,-val_ind]
    
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
    if(pen_on == 1){ymat_pen_opt = ymat_pen00[ind_opt,,]}
    if(pen_on == 0){ind_opt = 1;ymat_pen_opt = ymat_pen00[ind_opt,,]}
    
    ymat_QR = ymat_pen_opt
}

#### Evaluate the table ####
res_table = array(0, dim=c(4,2))
colnames(res_table) = c("Cov", "Width")
rownames(res_table) = c("QR","GCDS","GCDS_C","WGAN")


#### Evaluate the perfroamce on the out-of-sample test data ####
for(i in 1:nrow(Xtt)){
    library(HDInterval)
    yt_QR = ymat_QR[,i]
    if(fGAN == 1){
        mm0 = 2
        yt_fGAN = ymat_fGAN[,i]
        CI_pen = quantile(yt_fGAN, probs=c(0.025,0.975))
        if(ytt[i]>=CI_pen[1]&ytt[i]<CI_pen[2]){res_table[mm0,1]=res_table[mm0,1]+1}
        res_table[mm0,2]=res_table[mm0,2]+abs(CI_pen[1]-CI_pen[2])}
    if(WGAN == 1){
        mm0 = 4
        yt_WGAN = ymat_WGAN[,i]
        CI_pen = quantile(yt_WGAN, probs=c(0.025,0.975))
        if(ytt[i]>=CI_pen[1]&ytt[i]<CI_pen[2]){res_table[mm0,1]=res_table[mm0,1]+1}
        res_table[mm0,2]=res_table[mm0,2]+abs(CI_pen[1]-CI_pen[2])}
    if(fGAN_C == 1){
        mm0 = 3
        yt_fGAN_C = ymat_fGAN_C[,i]
        CI_pen = quantile(yt_fGAN_C, probs=c(0.025,0.975))
        if(ytt[i]>=CI_pen[1]&ytt[i]<CI_pen[2]){res_table[mm0,1]=res_table[mm0,1]+1}
        res_table[mm0,2]=res_table[mm0,2]+abs(CI_pen[1]-CI_pen[2])}
    
    if(method == "QR"|method == "QR_m"){
        quant = quantile(yt_QR, probs=c(0.02,0.98))
        if(model_type == "fish"){quant = quantile(yt_QR, probs=c(0.04,0.96))}
        if(model_type == "noise"){quant = quantile(yt_QR, probs=c(0.0004,0.9996))}
        #quant = quantile(yt_QR, probs=c(0.0004,0.9996))
        ind_out = c(which(yt_QR>=quant[2]), which(yt_QR<=quant[1]))
        yt_QR = yt_QR[-ind_out]
        n0 = length(yt_QR)
    } 
    
    #CI_QRpen = hdi(yt_gen_pen, credMass=0.95)
    CI_QRpen = quantile(yt_QR, probs=c(0.025,0.975))
    CI_table[i,1:2] = c(Xtt[i], ytt[i])
    CI_table[i,3:4] = CI_QRpen
    #QR CI
    if(ytt[i]>=CI_QRpen[1]&ytt[i]<CI_QRpen[2]){res_table[1,1]=res_table[1,1]+1}
    res_table[1,2]=res_table[1,2]+abs(CI_QRpen[1]-CI_QRpen[2])
    
    #Flexcode CI
    if(flex == 1){
        #Flexcode CI
        rr = flex_cde[i,]-flex_thres[i,]+1e-20
        rr1 = rr[-1];rr1[length(rr)] = rr[length(rr)]
        r = rr*rr1
        root_ind = which(r<=0)+1
        CI_pen = flex_grid[root_ind]
        mm0 = 4
        if(length(CI_pen) == 1){
            CI_pen0 = c(CI_pen, flex_grid[1])
            CI_pen00 = c(CI_pen, flex_grid[length(r)])
            if(diff(CI_pen0)>=diff(CI_pen00)){CI_pen = CI_pen0}
            if(diff(CI_pen0)<diff(CI_pen00)){CI_pen = CI_pen00}}
        if(length(CI_pen)%%2 !=0){
            max_ind = which.max(diff(CI_pen))
            CI_pen = CI_pen[max_ind:(max_ind+1)]}
        if(length(CI_pen) == 2){
            if(yt[i]>=CI_pen[1]&yt[i]<CI_pen[2]){
                res_table[mm0,1]=res_table[mm0,1]+1}
            res_table[mm0,2]=res_table[mm0,2]+abs(CI_pen[1]-CI_pen[2])}
        else{
            if(yt[i]>=CI_pen[1]&yt[i]<CI_pen[2]){res_table[mm0,1]=res_table[mm0,1]+1}
            if(yt[i]>=CI_pen[3]&yt[i]<CI_pen[4]){res_table[mm0,1]=res_table[mm0,1]+1}
            res_table[mm0,2]=res_table[mm0,2]+abs(CI_pen[1]-CI_pen[2]+CI_pen[3]-CI_pen[4])}}
}

#### print out the results ####
res_table = round(res_table/nrow(Xtt),4)
print(res_table)

