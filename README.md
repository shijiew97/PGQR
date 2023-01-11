# PGQR
This is code to implement the Penalized Generative Quantile Regression (PGQR) model in "Generative Quantile Regression with Variability Penalty" by Shijie Wang, Minsuk Shin, and Ray Bai. https://arxiv.org/abs/2301.03661

## Abstract 
We introduce a deep learning generative model for joint quantile estimation called Penalized Generative Quantile Regression (PGQR). Our approach simultaneously generates samples from many random quantile levels, allowing us to infer the conditional distribution of a response variable given a set of covariates. Our method employs a novel variability penalty to avoid the problem of vanishing variability, or memorization, in deep generative models. Further, we introduce a new family of partial monotonic neural networks (PMNN) to circumvent the problem of crossing quantile curves. A major benefit of PGQR is that it can be fit using a single optimization, thus bypassing the need to repeatedly train the model at multiple quantile levels or use computationally expensive cross-validation to tune the penalty parameter. We illustrate the efficacy of PGQR through extensive simulation studies and analysis of real datasets.

## Prerequisites for PGQR
In order to sucessfully run the PGQR model, we need to pre-install and confirm the following environments on your local machine. Moreover, there are several R packages that need to be installed beforehand. 

#### _A_. Python, Pytorch and CUDA environment
The main code for implementing PGQR is in `Python` while simulation and data generation is coded in `R`. The partial monotonic neural networks (PMNN) is constructed by the __Pytorch__ library. We strongly recommend using `CUDA` (GPU-based tool) to train PGQR, which can accelerate the runtime a lot more than using `CPU`.
- __Python__ 3.7 or above
- __[Pytroch](https://pytorch.org/)__ 1.11.0 or above
- __[NAVID CUDA](https://developer.nvidia.com/cuda-toolkit)__ 10.2 or above

#### _B_. Required R package 
In `R`, we need the `reticulate` package to run PGQR which is coded in `Python` in R. For comparison, we also considered other traditional CDE methods, including 
- _Random Forest CDE_ (RFCDE)
- _Nearest Neighbor Conditional Density Estimation_ (NNKCDE)
- [FlexCoDE](https://github.com/rizbicki/FlexCoDE) the specifics of FlexCoDE installation can be found in at [FlexCoDE](https://github.com/rizbicki/FlexCoDE). 

The motorcycle dataset is included in the `adlift` package and nonparameteric quantile regression is implemented using R package `quantreg`.
```
install.package("reticulate")
install.package("RFCDE")
install.package("NNKCDE")
install.package("HDInterval")
install.package("adlift")
install.package("quantreg")
```

# Implementation of PGQR
To implement PGQR, we provide the `Python` code of PGQR under _Python code_ folder and `R` code for simulation under _R code_ folder. More detailed expalanations are provided below.

## Working directory  ####
To run the PGQR model, save the results and produce the plots, we need to set the working directory beforehand. _It's very crucial to set the working directory manually "/yourlocalmachine/" to sucessfully run the PGQR model for __every R code file__ ._

- Create a Python code folder: "/yourlocalmachine/Python_code/" which should inculdes the python files the same in 'Github/Python_code/' folder.
- Create a R code folder: "/yourlocalmachine/R_code/" which should inculdes the R files the same in 'Github/R_code/' folder.
- Create a result folder: "/yourlocalmachine/result/" 
- Create two subfolders in the result folder: "/yourlocalmachine/result/2000/" for simulation studies and "/yourlocalmachine/result/real/" for real data analysis.  

## __Python code__ folder
Under the _Python_code_ folder, we have the following scripts:
- __QR_pen_m.py__ constructs the main body of _penalized Generative Quantile Regression_ (PGQR).
- __QR_nopen_m.py__ constructs the _Generative Quantile Regression_(GQR) without regularization term.
- __Cond_WGAN.py__ constructs the Wasserstein generative conditional sampler ([WGCS](https://arxiv.org/pdf/2112.10039.pdf)).
- __CondGAN_MS.py__ constructs Generative conditional distribution sampler ([GCDS](https://www.tandfonline.com/doi/abs/10.1080/01621459.2021.2016424)).

## __R_code__ folder
_It is very crucial to set the working directory manually such as "/yourlocalmachine/" to sucessfully run the PGQR model for __every R code file__ ._ Under the _R_code_ folder, we provide code for training PGQR, saving the results in _.RData_ form, plotting the graphs present in paper, and contructing the results tables for the simulation studies and real data analyses in Section 6 of the manuscript.

#### _A_. PGQR Simulation Train

- __PGQR.R__ is to train PGQR model under different simulation settings (see code annotation and descriptions in paper). The results should be saved under the path "/result/2000" where "/2000/" is the corresponding sample size.
- __data_gen.R__ is to generate the simulation dataset. It should be under "/R_code/" directory.
- __model_fit.R__ is to run the Python code for the deep generative models such as PGQR, GCDS or WGCS.

#### _B_. PGQR Simulation Graph

- __graph.R__ is to plot the graph from the saved results (in .RData form). The resultant graph will be saved in "/yourlocalmahine/result/2000/"

#### _C_. PGQR Simulation table

- __table_compute.R__ is to compute the simulation table in the paper and save the corresponding results. 
- __table_eval.R__ is to evaluate the performance measure described in paper from the results saved by __table_compute.R__and summarize them in table form. 
- __quantile_plot.R__ is to plot the predicted mean squared error (PMSE) of different quantiles and produce the resultant plot, which is evaluated from __table_compute.R__.

#### _D_. Real data analysis

- __real_fit.R__ is to implement PGQR on three real datasets, as well as two classic crossing-quantile benchmark datasets and save the results under "/yourlocalmachine/real/"
- __real.table.R__ is to evaluate the out-of-sample prediction interval width and coverage rate from results produced by __real_fit.R__.
- __cross_quantile.R__ is to produce the plot of the crossing quantile phenomenon in the motorcycle and bone mass density datasets from the results by __real_fit.R__.
- __APL_plot.R__ is to produce the plot of APL/F ratio plot from the results by __real_fit.R__.

























