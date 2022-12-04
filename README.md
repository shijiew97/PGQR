# PGQR
This is an R package to implement the Penalized Generative Qunatile Regression (PGQR)

### Abstract 
We introduce a deep learning generative model for simultaneous quantile regression called Penalized Generative Quantile Regression (PGQR). Our approach simultaneously generates samples from a large number of random quantile levels, thus allowing us to infer the conditional density of a response variable given a set of covariates. Our method also employs a novel _variability penalty_ to avoid the common problem of vanishing variance in deep generative models. Furthermore, we introduce a new family of neural networks called partial monotonic neural networks (PMNN) to circumvent the problem of crossing quantile planes. A major benefit of PGQR is that our method can be fit using a single optimization, thus bypassing the need to repeatedly train the model at multiple quantile levels or use computationally expensive cross-validation to tune the penalty parameter. We illustrate the efficacy of PGQR through extensive simulation studies and analysis of real datasets.

### Pre-requisite of PGQR
In order to sucessfully run the PGQR model, we need to pre-install and confirm the following environment in local machine. The main of PGQR is coded in `Python` while simulation and data generation is coded in `R`. The partial monotonic neural networks (PMNN) is constructed by __Pytorch__ library and we strongly recommend using `CUDA` (GPU-Based tool) to train PGQR which can be accelerated a lot than using `CPU`.
- __Python__ 3.7 or above
- __[Pytroch](https://pytorch.org/)__ 1.11.0 or above
- __[NAVID CUDA](https://developer.nvidia.com/cuda-toolkit)__ 10.2 or above

In R, we also need `reticulate` package to run `Python` in R.
```
install.package("reticulate")
```

### Runing PGQR

