#### Setting the working directory ####
#reticulate::use_condaenv(condaenv='jackie', required=TRUE)
#rm(list=ls())
#setwd("~/Dropbox/Shijie/GR")
#setwd("C:/Users/18036/Dropbox/Shijie/GR")

#### Simulation setting ####
model_type = "takeuchi06"

#### Different model ####
method = "QR_m"                #PGQR
#method = "QR_nopen_m"         #GQR
#method = "fGAN"               #GCDS
#method = "fGAN_C"             #deep-GCDS
#method = "WGAN"               #WGCS

#### Real dataset ####
if(model_type == "takeuchi06"){
    set.seed(123456)
    n = 2000
    x = runif(n, -1, 1)
    Y = sin(pi*x) / (pi*x) + rnorm(n, mean=0, sd=0.1 * exp(1-x))
    X = as.matrix(x, ncol=1) 
    #Xt = as.matrix(seq(-1, 1, length=300), ncol=1)
    Xt = as.matrix(x, ncol=1)
    y = as.matrix(Y, ncol=1); yt = as.matrix(y, ncol=1)
    p = 1
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

lam_min = -60.0                #minimum of lambda candidate
lam_max = 3.0                  #maximum of lambda candidate

#n = 2000                       #sample size
n0 = n                          
zn = 100
ntest = 100
#p = 5.0                        #covariate dimension
if(model_type == "reg_P111"){p = 1.0}
S = n
fac = 1.0                      #alpha value


L = 2                          #number of hidden layers
batchnorm_on = 1
num_it = 10000*2               #number of iterations    
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

source("./R_code/model_fit.R")
#source("~/Dropbox/Shijie/GR/Github code/R Code/model_fit.R")

#### Save the results ####
save = 1
if(save == 1){gr_res = list(X, y, Xt, yt, ymat, discr, lam_cand0, sigma0, p)} 
if(pen_on == 0){data_type = paste(model_type,"_",method,sep="")}
if(pen_on == 1){data_type = paste(model_type,"_",method,"_pen",sep="")}

#### Save ####
save_path = paste(getwd(), "/result/real/", data_type, "_gr.RData", sep="")
saveRDS(gr_res, save_path)