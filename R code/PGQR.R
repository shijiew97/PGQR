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
source("./R code/data_gen.R")

#### Fit the PGQR model ####
if(method == "fGAN"){L=2;hidden_size=40;zn=3;pen_on=0}
if(method == "fGAN_C"){L=3;hidden_size=1000;zn=100;pen_on=0}
if(method == "WGAN"){L=1;hidden_size=40;zn=3;pen_on=0}
if(method == "QR_m"){pen_on=1}
if(method == "QR_nopen_m"){pen_on=0}

source("./R code/model_fit.R")

#### GAN example ####
if(pen_on == 0){data_type = paste(model_type,"_",method,sep="")}
if(pen_on == 1){data_type = paste(model_type,"_",method,"_pen",sep="")}

sensitive=0
if(sensitive==0){data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K", sep="")}
if(sensitive==1){data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K","_a_",fac,sep="")}

#### Save the results (if needed) ####
save = 1
if(save == 1){gr_res = list(X, y, Xt, yt, ymat, discr, lam_cand0, sigma0, p)}

#### Saving path ####
path0 = paste("/result/",n,"/",sep="")
save_path = paste(getwd(), path0, data_type, "_gr.RData", sep="")
saveRDS(gr_res, save_path)

