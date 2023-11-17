#### Setting the working directory ####
#rm(list=ls())
#setwd("~/Dropbox/Shijie/GR")
#setwd("C:/Users/18036/Dropbox/Shijie/GR")

sim_cand = c("reg_P111","reg_skew","reg_multimode","reg_nonparam","reg_linear","reg_norm")
method_cand = c('QR_m', 'l1-p', 'mcqrnn')
num_rep = 1*20

result = array(0, dim=c(3, length(sim_cand), 3))
sd = 0
if(sd == 1){stat_sd = array(0, dim=c(3, length(sim_cand), num_rep, 3))}

for(ii in 1:length(method_cand)){
    ##Which method to use
    method = method_cand[ii]
    ##parameter
    num_rep = 1*20                 #number of replication
    n_test = 100                   #number of out-of-sample test points
    tau_cand = seq(0.1, 0.9, length=9)
    #### Here: choose different alpha value ####
    n = 2000                       #sample size
    p = 5.0                        #covariate dimension
    #fac = 1.0
    for(jj in 1:length(sim_cand)){
        model_type = sim_cand[jj]
        if(method == 'QR_m'){
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
                lam_max = 0.1
                
                
                #n = 2000
                n0 = n
                zn = 100
                ntest = 100
                #p = 10/2
                S = n
                fac = 1.0
                if(model_type == 'reg_multimode'){fac=5.0}
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
                    
                    stat_gen = quantile(yt_gen_pen,probs=tau_cand)
                    ##Total deviance of quantile function
                    rk = c(0)
                    for(k in 1:9){
                        cur = sum( (y_gen <= stat_gen[k]) )/n0
                        rk = c(rk, cur)
                    }
                    rk = c(rk, 1)
                    pk = diff(rk)
                    tv = sum(abs(0.1 - pk))
                    hd = sum( (sqrt(0.1)-sqrt(pk))^2 )
                    kl = -sum( 0.1*log(pk) )
                    
                    result[ii, jj, 1] = result[ii, jj, 1] + tv/(num_rep*n_test)
                    result[ii, jj, 2] = result[ii, jj, 2] + hd/(num_rep*n_test)
                    result[ii, jj, 3] = result[ii, jj, 3] + kl/(num_rep*n_test)
                    
                    if(sd == 1){
                        stat_sd[ii, jj, j, 1] = stat_sd[ii, jj, j, 1] + tv/(n_test)
                        stat_sd[ii, jj, j, 2] = stat_sd[ii, jj, j, 2] + hd/(n_test)
                        stat_sd[ii, jj, j, 3] = stat_sd[ii, jj, j, 3] + kl/(n_test)}
                }
            }
        }
        else{
            p = 5.0
            sigma0 = 1
            n = 2000
            num_it = 2000
            
            if(model_type == "reg_P111"){p=1}
            if(model_type == "reg_linear"){p=1;sigma0=0.1}
            
            data_type = paste(model_type,"_",method,sep="")
            data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K", "_COMPARISON", sep="")
            
            path0 = paste("/result/",n,"/",sep="")
            save_path_tab = paste(getwd(), path0, data_type, "_gr_TABLE.RData", sep="")
            result[ii, jj, ] = readRDS(save_path_tab)
            
            if(sd == 1){
                res = readRDS(paste(getwd(), path0, data_type, "_gr.RData", sep=""))
                for(j in 1:num_rep){
                    #if(model_type != "reg_multimode"){Seed = Seed + j -1}
                    source("./R_code/data_gen.R")
                    n_test = 100
                    for(i in 101:200){
                        
                        X_gen = matrix(rep(Xt[i,], each=n0), n0, p)
                        Z_gen = z_true(X_gen,n0)
                        y_gen = gen_true(X_gen, Z_gen)
                        
                        #stat_gen = quantile(yt_gen_pen,probs=tau_cand)
                        stat_gen = res[j,i-100,]
                        ##Total deviance of quantile function
                        rk = c(0)
                        for(k in 1:9){
                            cur = sum( (y_gen <= stat_gen[k]) )/n0
                            rk = c(rk, cur)
                        }
                        rk = c(rk, 1)
                        pk = diff(rk)
                        tv = sum(abs(0.1 - pk))
                        hd = sum( (sqrt(0.1)-sqrt(pk))^2 )
                        kl = -sum( 0.1*log(pk) )
                        
                        stat_sd[ii, jj, j, 1] = stat_sd[ii, jj, j, 1] + tv/(n_test)
                        stat_sd[ii, jj, j, 2] = stat_sd[ii, jj, j, 2] + hd/(n_test)
                        stat_sd[ii, jj, j, 3] = stat_sd[ii, jj, j, 3] + kl/(n_test)}
                }
        }
}}}
print(round(result, 3))
#print(round(apply(stat_sd, c(1,2,4), sd), 3))

#### Visualization-barplot-all ####
#rm(list=ls())
#setwd("~/Dropbox/Shijie/GR")
#setwd("C:/Users/18036/Dropbox/Shijie/GR")

color <- c(rgb(0,0,1,0.8), rgb(1,0,0,0.8), rgb(0.5,0,0.5,0.5))
n = 2000
path0 = paste("/result/",n,"/",sep="")
png = 1
if(png == 1){
    name0 <- paste(getwd(), path0, "quantile_compare.png", sep="")
    png(name0, width=1450, height=400)
}else{
    name0 <- paste(getwd(), path0, "quantile_compare.pdf", sep="")
    pdf(name0, height=3, width=10)
}

par(mfrow=c(1,2), mai=c(0.8,1.0,0.6,0.6))

barplot((result[,,1]), beside=T, names.arg=paste("Simulation ", seq(1,6,1), sep=''), cex.names=1.2,
        col=color, ylim=c(0, 2), cex.lab=2.0, cex.axis=1.5, main="TV", cex.main=2.0,
        ylab="value")
barplot((result[,,2]), beside=T, names.arg=paste("Simulation ", seq(1,6,1), sep=''), cex.names=1.2,
        col=color, ylim=c(0, 2), cex.lab=0.5, cex.axis=1.5, main="HD", cex.main=2.0)
legend("topright", 
       c("PGQR", "NMQN", "MCQRNN"),
       fill=color,
       bty="n", cex=2.0,
       y.intersp=0.8)

dev.off()




#### Visualization-barplot-main ####
#rm(list=ls())
#setwd("~/Dropbox/Shijie/GR")
setwd("C:/Users/18036/Dropbox/Shijie/GR")

color <- c(rgb(0,0,1,0.8), rgb(1,0,0,0.8), rgb(0.5,0,0.5,0.5))
n = 2000
path0 = paste("/result/",n,"/",sep="")
png = 1
if(png == 1){
    name0 <- paste(getwd(), path0, "quantile_compare_main.png", sep="")
    png(name0, width=1450, height=400)
}else{
    name0 <- paste(getwd(), path0, "quantile_compare_main.pdf", sep="")
    pdf(name0, height=3, width=10)
}

par(mfrow=c(1,2), mai=c(0.8,1.0,0.6,0.6))

barplot((result[,1:3,1]), beside=T, names.arg=paste("Simulation ", seq(1,3,1), sep=''), cex.names=1.85,
        col=color, ylim=c(0, 1.5), cex.lab=2.0, cex.axis=1.5, main="TV", cex.main=2.0,
        ylab="value")
barplot((result[,1:3,2]), beside=T, names.arg=paste("Simulation ", seq(1,3,1), sep=''), cex.names=1.85,
        col=color, ylim=c(0, 1.5), cex.lab=0.5, cex.axis=1.5, main="HD", cex.main=2.0)
legend("topright", 
       c("PGQR", "NMQN", "MCQRNN"),
       fill=color,
       bty="n", cex=2.0,
       y.intersp=0.8)

dev.off()
#### Visualization-barplot-appedix ####
#rm(list=ls())
#setwd("~/Dropbox/Shijie/GR")
setwd("C:/Users/18036/Dropbox/Shijie/GR")

color <- c(rgb(0,0,1,0.8), rgb(1,0,0,0.8), rgb(0.5,0,0.5,0.5))
n = 2000
path0 = paste("/result/",n,"/",sep="")
png = 1
if(png == 1){
    name0 <- paste(getwd(), path0, "quantile_compare_ap.png", sep="")
    png(name0, width=1450, height=400)
}else{
    name0 <- paste(getwd(), path0, "quantile_compare_ap.pdf", sep="")
    pdf(name0, height=3, width=10)
}

par(mfrow=c(1,2), mai=c(0.8,1.0,0.6,0.6))

barplot((result[,4:6,1]), beside=T, names.arg=paste("Simulation ", seq(4,6,1), sep=''), cex.names=1.85,
        col=color, ylim=c(0, 2), cex.lab=2.0, cex.axis=1.5, main="TV", cex.main=2.0,
        ylab="value")
barplot((result[,4:6,2]), beside=T, names.arg=paste("Simulation ", seq(4,6,1), sep=''), cex.names=1.85,
        col=color, ylim=c(0, 2), cex.lab=0.5, cex.axis=1.5, main="HD", cex.main=2.0)
legend("topright", 
       c("PGQR", "NMQN", "MCQRNN"),
       fill=color,
       bty="n", cex=2.0,
       y.intersp=0.8)

dev.off()