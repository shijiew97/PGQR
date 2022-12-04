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

#### Real dataset ####
if(model_type == "noise"){
    dat = data.frame(read.table("./r code/noise.dat", skip=3))
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
    Xt = as.matrix(dat_test[,1:p]);yt = as.matrix(dat_test[,p+1])}
if(model_type == "machine"){
    
    dat = data.frame(read.csv("./r code/machine.csv",sep=";"))
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
    Xt = as.matrix(dat_test[,2:p]);yt = as.matrix(dat_test[,1])}
if(model_type == "fish"){
    
    dat = data.frame(read.csv("./r code/fish.csv", sep=";"))
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
}
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
#source("./R code/data_gen.R")

#### Fit the PGQR model ####
if(method == "fGAN"){L=2;hidden_size=40;zn=3;pen_on=0}
if(method == "fGAN_C"){L=3;hidden_size=1000;zn=100;pen_on=0}
if(method == "WGAN"){L=1;hidden_size=40;zn=3;pen_on=0}
if(method == "QR_m"){pen_on=1}
if(method == "QR_nopen_m"){pen_on=0}

source("./R code/model_fit.R")


#### Save the results ####
save = 1
if(save == 1){gr_res = list(X, y, Xt, yt, ymat, discr, lam_cand0, sigma0, p)} 
if(pen_on == 0){data_type = paste(model_type,"_",method,sep="")}
if(pen_on == 1){data_type = paste(model_type,"_",method,"_pen",sep="")}

#### Save ####
save_path = paste(getwd(), "/result/real/", data_type, "_gr.RData", sep="")
saveRDS(gr_res, save_path)