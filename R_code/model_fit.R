#### GR model fitting ####
library(reticulate)

#### Transforming procedure ####
y1 = r_to_py(y, convert=FALSE)
X1 = r_to_py(X, convert=FALSE)
Xt1 = r_to_py(Xt, convert=FALSE)
yt1 = r_to_py(yt, convert=FALSE)

S1 = r_to_py(S, convert=FALSE)
n01 = r_to_py(n0, convert=FALSE)

hidden_size1 = r_to_py(hidden_size, convert=FALSE)
L1 = r_to_py(L); NN_type = r_to_py(NN_type, convert=FALSE)
n1 = r_to_py(n, convert=FALSE)
p1 = r_to_py(p, convert=FALSE)
lr01 = r_to_py(lr, convert=FALSE)
gpu_ind1 = r_to_py(gpu_ind, convert=FALSE)
verb1 = r_to_py(verb, convert=FALSE)
num_it1 = r_to_py(num_it, convert=FALSE)
batchnorm_on1 = r_to_py(batchnorm_on, convert=FALSE)

zn1 = r_to_py(zn, convert=FALSE)
m1 = r_to_py(m, convert=FALSE)
lrdecay1 = r_to_py(lrdecay, convert=FALSE)
NN_type1 = r_to_py(NN_type, convert=FALSE)
lr_power1 = r_to_py(lr_power, convert=FALSE)
boot_size1 = r_to_py(boot_size, convert=FALSE)
test1 = r_to_py(test, convert=FALSE)
l1pen1 = r_to_py(l1pen, convert=FALSE)
N1 = r_to_py(N, convert=FALSE)
K1 = r_to_py(k, convert=FALSE)
fac1 = r_to_py(fac, convert=FALSE)
lam_max1 = r_to_py(lam_max, convert=FALSE)
lam_min1 = r_to_py(lam_min, convert=FALSE)



#### Code path ####
if(is.null(pen_on) == FALSE){
    #### GR example ####
    library(reticulate)
    
    #### Transforming procedure ####
    y1 = r_to_py(y, convert=FALSE)
    X1 = r_to_py(X, convert=FALSE)
    Xt1 = r_to_py(Xt, convert=FALSE)
    yt1 = r_to_py(yt, convert=FALSE)
    
    S1 = r_to_py(S, convert=FALSE)
    n01 = r_to_py(n0, convert=FALSE)
    
    hidden_size1 = r_to_py(hidden_size, convert=FALSE)
    L1 = r_to_py(L); NN_type = r_to_py(NN_type, convert=FALSE)
    n1 = r_to_py(n, convert=FALSE)
    p1 = r_to_py(p, convert=FALSE)
    lr01 = r_to_py(lr, convert=FALSE)
    gpu_ind1 = r_to_py(gpu_ind, convert=FALSE)
    verb1 = r_to_py(verb, convert=FALSE)
    num_it1 = r_to_py(num_it, convert=FALSE)
    batchnorm_on1 = r_to_py(batchnorm_on, convert=FALSE)
    zn1 = r_to_py(zn, convert=FALSE)
    m1 = r_to_py(m, convert=FALSE)
    lrdecay1 = r_to_py(lrdecay, convert=FALSE)
    NN_type1 = r_to_py(NN_type, convert=FALSE)
    lr_power1 = r_to_py(lr_power, convert=FALSE)
    boot_size1 = r_to_py(boot_size, convert=FALSE)
    test1 = r_to_py(test, convert=FALSE)
    l1pen1 = r_to_py(l1pen, convert=FALSE)
    N1 = r_to_py(N, convert=FALSE)
    K1 = r_to_py(k, convert=FALSE)
    fac1 = r_to_py(fac, convert=FALSE)
    lam_max1 = r_to_py(lam_max, convert=FALSE)
    lam_min1 = r_to_py(lam_min, convert=FALSE)
    pen_on1 = r_to_py(pen_on, convert=FALSE)
    
    
    #fGan 
    Code_GBR2 = "./Python_code/CondGAN_MS.py"
    reticulate::source_python(Code_GBR2, convert = FALSE, envir = globalenv())
    
    #WGAN
    Code_GBR2 = "./Python_code/Cond_WGAN.py"
    reticulate::source_python(Code_GBR2, convert = FALSE, envir = globalenv())
    
    #PGQR
    Code_GBR2 = "./Python_code/QR_pen_m.py"
    reticulate::source_python(Code_GBR2, convert = FALSE, envir = globalenv())
    
    #GQR without varibility penalty
    Code_GBR2 = "./Python_code/QR_nopen_m.py"
    reticulate::source_python(Code_GBR2, convert = FALSE, envir = globalenv())
    


    if(method == "fGAN"){
        fit_GBR2 = CondGAN(y1, X1, Xt1, hidden_size1, gpu_ind1, 
                           NN_type1, L1, S1, n1, zn1, p1, batchnorm_on1, 
                           num_it1, lr01, lrdecay1, lr_power1, m1, verb1, 
                           boot_size1, test1, l1pen1, N1, K1, yt1, fac1, 
                           lam_min1, lam_max1, n01, pen_on1)
    }
    if(method == "fGAN_C"){
        fit_GBR2 = CondGAN(y1, X1, Xt1, hidden_size1, gpu_ind1, 
                           NN_type1, L1, S1, n1, zn1, p1, batchnorm_on1, 
                           num_it1, lr01, lrdecay1, lr_power1, m1, verb1, 
                           boot_size1, test1, l1pen1, N1, K1, yt1, fac1, 
                           lam_min1, lam_max1, n01, pen_on1)
    }
    if(method == "WGAN"){
        fit_GBR2 = CondWGAN(y1, X1, Xt1, hidden_size1, gpu_ind1, 
                            NN_type1, L1, S1, n1, zn1, p1, batchnorm_on1, 
                            num_it1, lr01, lrdecay1, lr_power1, m1, verb1, 
                            boot_size1, test1, l1pen1, N1, K1, yt1, fac1, 
                            lam_min1, lam_max1, n01, pen_on1)
    }
    if(method == "QR_m"){
        fit_GBR2 = QR_pen_m(y1, X1, Xt1, hidden_size1, gpu_ind1, 
                            NN_type1, L1, S1, n1, zn1, p1, batchnorm_on1, 
                            num_it1, lr01, lrdecay1, lr_power1, m1, verb1, 
                            boot_size1, test1, l1pen1, N1, K1, yt1, fac1, 
                            lam_min1, lam_max1, n01, pen_on1)
    }
    if(method == "QR_nopen_m"){
        fit_GBR2 = QR_nopen_m(y1, X1, Xt1, hidden_size1, gpu_ind1, 
                              NN_type1, L1, S1, n1, zn1, p1, batchnorm_on1, 
                              num_it1, lr01, lrdecay1, lr_power1, m1, verb1, 
                              boot_size1, test1, l1pen1, N1, K1, yt1, fac1, 
                              lam_min1, lam_max1, n01, pen_on1)
    }

    ymat = py_to_r(fit_GBR2)[[1]]
    discr = py_to_r(fit_GBR2)[[2]]
    lam_cand0 = py_to_r(fit_GBR2)[[3]]
    
}


