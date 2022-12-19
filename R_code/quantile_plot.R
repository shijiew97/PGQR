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


#### Graph for Quantile Difference: PGQR #### 
num_plot = 3
n = 2000
num_plot = 2
data_type = "Simulation"
path0 = paste("/result/",n,"/",sep="")
name0 = paste(getwd(), path0, data_type, "_Quantile_Difference.png", sep="")
if(num_plot == 2){name0 = paste(getwd(), path0, data_type, "_Quantile_Difference2.png", sep="")}

#### Graph setting ####
if(num_plot == 9){
    png(name0, width=850, height=500)
    par(mfrow=c(3,3), mai=c(0.4,0.4,0.2,0.2))
}
if(num_plot == 6){
    png(name0, width=850, height=350)
    par(mfrow=c(2,3), mai=c(0.4,0.4,0.2,0.2))
}
if(num_plot == 3){
    png(name0, width=1050, height=300)
    par(mfrow=c(1,3), mai=c(0.6,0.6,0.4,0.4))
}
if(num_plot == 2){
    png(name0, width=1050, height=400)
    par(mfrow=c(1,2), mai=c(0.8,0.8,0.6,0.6))
}

#### Graph plot simulation 1-3 ####
for(model_type in c("reg_P111", "reg_skew","reg_multimode")){
    
    ##read results
    read00 = 1
    if(read00 == 1){
        p = 5
        n = 2000
        num_it = 5000*2
        sigma0 = 1.0
        Seed = 128783
        source("./R_code/data_gen.R")
        
        pen_on = 1
        
        if(pen_on == 0){data_type = paste(model_type,"_",method,sep="")}
        if(pen_on == 1){data_type = paste(model_type,"_",method,"_pen",sep="")}
        
        data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K", "_TABLE", sep="")
        
        path0 = paste("/result/",n,"/",sep="")
        read_path = paste(getwd(), path0, data_type, "_gr.RData", sep="")
        tab = round(readRDS(read_path),3)
    }
    
    ##get plot
    tau =  seq(0.1, 0.9, length=9)
    quan_v = tab[,3:11]
    method_name = c("GCDS","deep-GCDS","WCGS","PGQR")
    plt_name = c("S1", "S2", "S3")
    if(model_type == "reg_skew"){
        Ylim = c(0,15);j=1
    }
    if(model_type == "reg_multimode"){
        Ylim = c(0,15);j=2
    }
    if(model_type == "reg_P111"){
        Ylim = c(0,400);j=3
    }
    plot(x=tau, y=quan_v[1,], type="b", pch=16, col=1, ylim=Ylim,
         xlab="", ylab="", xaxt="n", lty=2, cex.axis=1.2, cex=2.0, lwd=2.5)
    axis(1, at=tau,  cex.axis=1.2)
    for(i in 2:4){
        lines(x=tau, y=quan_v[i,], col=i, type="b", pch=i, lty=2, cex=2.0, lwd=2.5)
        if(i==4){lines(x=tau, y=quan_v[i,], col=i, type="b", pch=i, lwd=2.5, cex=2.0)}
    }
    if(model_type=="reg_P111"){
        title(main="Simulation 1", family="mono", line=1.0, cex.main=2.0)
        title(ylab="Quantile PMSE", family="mono", line=2.5, cex.lab=2.0)
    }
    if(model_type=="reg_skew"){title(main="Simulation 2", family="mono", line=1.0, cex.main=2.0)}
    if(model_type=="reg_multimode"){title(main="Simulation 1", family="mono", line=1.0, cex.main=2.0)}
    title(xlab="Quantile", family="mono", cex.lab=1.7, line=2.5)
}
legend("topright", legend=method_name, col=seq(1:4), pch=c(16,2,3,4), lwd=seq(2.5,4),
       lty=c(2,2,2,2), bty="n", cex=1.5)
dev.off()

#### Graph plot simulation 4-5 ####
for(model_type in c("reg_nonparam","reg_linear")){
    
    ##read results
    read00 = 1
    if(read00 == 1){
        p = 5
        n = 2000
        num_it = 5000*2
        sigma0 = 1.0
        Seed = 128783
        if(model_type == "reg_linear"){p = 1.0;sigma0 = 0.1}
        source("./r code/GR_data_gen.R")
        
        pen_on = 1
        
        if(pen_on == 0){data_type = paste(model_type,"_",method,sep="")}
        if(pen_on == 1){data_type = paste(model_type,"_",method,"_pen",sep="")}
        
        data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K", "_TABLE", sep="")
        
        path0 = paste("/result/",n,"/",sep="")
        read_path = paste(getwd(), path0, data_type, "_gr.RData", sep="")
        tab = round(readRDS(read_path),3)
    }
    
    ##get plot
    tau =  seq(0.1, 0.9, length=9)
    quan_v = tab[,3:11]
    method_name = c("GCDS","deep-GCDS","WCGS","PGQR")
    #plt_name = c("S1", "S2", "S3")
    if(model_type == "reg_nonparam"){
        Ylim = c(0,1.5);j=1
    }
    if(model_type == "reg_linear"){
        Ylim = c(0,0.01);j=2
    }
    plot(x=tau, y=quan_v[1,], type="b", pch=16, col=1, ylim=Ylim,
         xlab="", ylab="", xaxt="n", lty=2, cex.axis=1.2, cex=2.0, lwd=2.5)
    axis(1, at=tau,  cex.axis=1.2)
    for(i in 2:4){
        lines(x=tau, y=quan_v[i,], col=i, type="b", pch=i, lty=2, cex=2.0, lwd=2.5)
        if(i==4){lines(x=tau, y=quan_v[i,], col=i, type="b", pch=i, lwd=2.5, cex=2.0)}
    }
    if(model_type=="reg_nonparam"){
        title(main="Simulation 4", family="mono", line=1.0, cex.main=2.0)
        title(ylab="Quantile PMSE", family="mono", line=2.5, cex.lab=2.0)
    }
    if(model_type=="reg_linear"){title(main="Simulation 5", family="mono", line=1.0, cex.main=2.0)}
    #if(model_type=="reg_multimode"){title(main="Simulation 3", family="mono", line=1.0, cex.main=2.0)}
    title(xlab="Quantile", family="mono", cex.lab=1.7, line=2.5)
}
legend("topright", legend=method_name, col=seq(1:4), pch=c(16,2,3,4), lwd=seq(2.5,4),
       lty=c(2,2,2,2), bty="n", cex=1.5)
dev.off()
print(name0)