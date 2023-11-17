### required package
#reticulate::use_condaenv(condaenv = 'tf')
#require(tensorflow)
#require(l1pm)
###############################################


#### Setting the working directory ####
#rm(list=ls())
#setwd("~/Dropbox/Shijie/GR")


#### Simulation setting ####
#model_type = "reg_P111"       #Simualtion 1
#model_type = "reg_nonparam"   #Simualtion 4
#model_type = "reg_simple"     #Overfitting illustration
#model_type = "reg_skew"       #Simulation 2
#model_type = "reg_linear"     #Simulation 5 (Small variance)
#model_type = "reg_multimode"  #Simulation 3
#model_type = "reg_norm"       #Simulation 6

##Simulation comparison
sim_cand = c("reg_P111","reg_skew","reg_multimode","reg_nonparam","reg_linear","reg_norm")

#### Calculate the MSE of median, mean, sd, quantile ##
evals = c("TV", "HD", "KL")
res_table = matrix(0, nrow=1, ncol=length(evals))
colnames(res_table) = evals

num_rep = 1*20                 #number of replication
n_test = 100                   #number of out-of-sample test points
tau_cand = seq(0.1, 0.9, length=9)

#### Here: choose different alpha value ####
fac = 1.0                      #alpha value
n = 2000                       #sample size
p = 5.0                        #covariate dimension

#### Specially for quantile models ####
method = 'l1-p'
#method = 'mcqrnn'

#### Save the result ##
num_rep = 1*20                 #number of replication


#### Evaluation of replication ##
for(sim in sim_cand){
    cat('Current simulation is, :', sim)
    model_type = sim
    ##table results
    evals = c("TV", "KL", "HD")
    res_table = matrix(0, nrow=1, ncol=length(evals))
    colnames(res_table) = evals
    ##all quantile estimates
    res = array(0, dim=c(num_rep,n_test,length(tau_cand)))
    ##seed
    Seed = 128783
    sigma0 = 1
    p = 5.0
    n = 2000
    
    if(model_type == "reg_linear"){p = 1.0;sigma0 = 0.1;lam_min = -60}
    if(model_type == "reg_P111"){p = 1.0}
    
    ##replication
    for(kk in 1:num_rep){
        cat('|', kk)
        if(model_type == "reg_multimode" | model_type == "reg_linear"){Seed = Seed}
        else{Seed = Seed + kk -1}
        source("./R_code/data_gen.R")
        num_it = 2000
        
        if(method == 'l1-p'){
            fit_result = l1_p(X = X,
                              y = y,
                              test_X = as.matrix(Xt[101:200,], ncol=p),
                              valid_X = as.matrix(Xt[1:100,], ncol=p),
                              tau = tau_cand,
                              hidden_dim1 = 4,
                              hidden_dim2 = 4,
                              learning_rate = 0.005,
                              max_deep_iter = num_it,
                              penalty = 0,
                              lambda_obj = 5)
            res[kk,,] = fit_result$y_test_predict
        }
        if(method == 'mcqrnn'){
            tmp_mcqrnn_fit = mcqrnn.fit(X, y, tau=tau_cand,
                                        n.hidden=4, n.hidden2=4, 
                                        n.trials=1, 
                                        iter.max=num_it, 
                                        penalty=5)
            res[kk,,] = mcqrnn.predict(Xt, tmp_mcqrnn_fit)
        }

        n_test = 100
        n0 = 2000
        for(i in 101:200){
            
            ##Samples from conditional distirbution
            X_gen = matrix(rep(Xt[i,], each=n0), n0, p)
            Z_gen = z_true(X_gen,n0)
            y_gen = gen_true(X_gen, Z_gen)
            
            ##results
            stat_gen = res[kk,i-100,]
            
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
            
            ##save results
            res_table[1, 1] = res_table[1, 1] + tv
            res_table[1, 2] = res_table[1, 2] + hd
            res_table[1, 3] = res_table[1, 3] + kl
        }
    }
    res_table = round(res_table/(num_rep*n_test),4)
    
    ### Save path ####
    data_type = paste(model_type,"_",method,sep="")
    data_type = paste(data_type,"_",p,"_", n, "_", num_it/1000, "K", "_COMPARISON", sep="")
    path0 = paste("/result/",n,"/",sep="")
    save_path = paste(getwd(), path0, data_type, "_gr.RData", sep="")
    save_path_tab = paste(getwd(), path0, data_type, "_gr_TABLE.RData", sep="")
    saveRDS(res, save_path)
    saveRDS(res_table, save_path_tab)
}
