
source("/Users/madhusreechowdhury/Desktop/Project_2_Revised 2/codes/fitted_values_logvMEM_new.R")

# Load libraries
library(dplyr)
library(ACDm)
library(doParallel)
library(nloptr)
library(Rsolnp)
library(doSNOW)

# Start time:
start_time <- Sys.time()

# Setting working directory:
setwd("/Users/madhusreechowdhury/Desktop/Project_2_Revised 2/codes")

# Source codes:
source("sparse_logvMEM_estimation_new.R")

#############################
# Componentwise:
#############################

# Setting random seed for reproducibility:
set.seed(123)

#####################################
# Step 1: Generating simulated data:
#####################################
d <- 5
p <- 10
q <- 0
r <- max(p,q)

# Max-lag matrices:
L <- matrix(c(rep(3,d),rep(4,d),rep(4,d),rep(5,d),rep(5,d)),nrow=d,ncol=d,byrow = TRUE) # HLag- componentwise

# List of A matrices:
A_matrix_list <- autoregressive_matrix(d = d,p = p,L = L,max_eigen_abs = 0.96)$A_matrix_list

# A matrix:
A_matrix <- do.call(cbind,A_matrix_list)

# Data: (Generating 1500 train data points and 1 for testing)
X <- sim_logvMEM(N = 1501,d = d,omega=matrix(rep(0.1,d),ncol=1),A=A_matrix_list
                 ,B=NULL,beta_par=4,burn_in = 500)

# Summary of X:
apply(X,2,mean)

# Covariance matrix:
var(X)

# Correlation matrix:
cor(X,method = "kendall")


################################
# Elementwise:
################################

# # Setting random seed for reproducibility:
# set.seed(123)
# 
# #####################################
# # Step 1: Generating simulated data:
# #####################################
# d <- 5
# p <- 10
# q <- 0
# r <- max(p,q)
# 
# # Max-lag matrices:
# # Lag-matrix:
# L <- matrix(c(3,0,0,3,4
#               ,2,4,2,2,4
#               ,3,3,5,4,4
#               ,2,2,3,5,3
#               ,4,4,4,4,6),nrow=d,ncol=d,byrow = TRUE) # HLag- elementwise
# 
# # List of A matrices:
# A_matrix_list <- autoregressive_matrix(d = d,p = p,L = L,max_eigen_abs = 0.96)$A_matrix_list
# 
# # A matrix:
# A_matrix <- do.call(cbind,A_matrix_list)
# 
# # Data: (Generating 1500 train data points and 1 for testing) 
# X <- sim_logvMEM(N = 1501,d = d,omega=matrix(rep(0.1,d),ncol=1),A=A_matrix_list
#                  ,B=NULL,beta_par=4,burn_in = 500)
# 
# # Summary of X:
# apply(X,2,mean)
# 
# # Covariance matrix:
# var(X)
# 
# # Correlation matrix:
# cor(X,method = "kendall")

# #########################
# # Own-Other
# #########################
# # Load libraries
# library(dplyr)
# library(ACDm)
# library(doParallel)
# library(nloptr)
# library(Rsolnp)
# library(doSNOW)
# 
# # Start time:
# start_time <- Sys.time()
# 
# # Setting working directory:
# setwd("/Users/madhusreechowdhury/Desktop/Project_2_Revised 2/codes/")
# 
# # Source codes:
# source("sparse_logvMEM_estimation_new.R")
# source("autoregressive_matrix_diag_dom.R")
# 
# # Setting random seed for reproducibility:
# set.seed(123)
# 
# #####################################
# # Step 1: Generating simulated data:
# #####################################
# d <- 5
# p <- 10
# q <- 0
# r <- max(p,q)
# 
# # Max-lag matrices:
# L <- matrix(c(6,6,5,5,5,
#               5,5,5,4,4,
#               4,3,4,4,4,
#               6,6,7,7,7,
#               6,5,7,4,7),byrow=TRUE,nrow=d,ncol=d) # Hlag-own-other
# 
# # List of A matrices:
# A_matrix_list <- autoregressive_matrix_own_other_diag_dom(d = d,p = p,L = L,max_eigen_abs = 0.96)$A_matrix_list
# 
# # A matrix:
# A_matrix <- do.call(cbind,A_matrix_list)
# 
# # Data: (Generating 1500 train data points and 1 for testing)
# X <- sim_logvMEM(N = 1501,d = d,omega=matrix(rep(0.1,d),ncol=1),A=A_matrix_list
#                  ,B=NULL,beta_par=4,burn_in = 500)
# 
# # Summary of X:
# apply(X,2,mean)
# 
# # Covariance matrix:
# var(X)
# 
# # Correlation matrix:
# cor(X,method = "kendall")


# Parameter matrix:
pardf <- read.csv("/Users/madhusreechowdhury/Desktop/Project_2_Revised 2/simulation_study/simulation_study_output/final_estimate-componentwise-group-scad.csv")

X <- X[1:1500,]
#i.index=1
par(mfrow=c(3,2))
for(i.index in 1:5){
  A_matrix_i = as.numeric(pardf[i.index,5:54])
  beta_par = unique(pardf[,"beta"])
  fit1 = fitted_values_logvMEM_i(X,i.index,A_matrix_i,beta_par,p,q)
  plot.ts(X[,i.index],col="red",main="Actual (red) vs Fitted (blue)")
  lines(fit1,col="blue")
}
