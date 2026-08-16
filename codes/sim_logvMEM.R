
# Seed:
#set.seed(1234)

# Source files:
#source("C:/Users/chira/Documents/GitHub/Project-2/core_files/multivariate_gamma_dist.R")
#source("C:/Users/chira/Documents/GitHub/Project-2/core_files/generate_autoregressive_matrices.R")

##################################################
# Custom functions: Generate lags of a matrix:
##################################################
generate_lag_matrix <- function(x_t,t,lag_order){
  mat_list <- list()
  for(i in 1:lag_order){
    mat_list <- append(mat_list,list(x_t[t-i,]))
  }
  return(mat_list)
}

#generate_lag_matrix(x_t=X,t=10,lag_order=1)

###############################################
# Custom functions: Sum product of two lists:
###############################################
list.sum.product <- function(list1,list2){
  a <- 0
  if(length(list1) == length(list2)){
    for(i in 1:length(list1)){
      a <- a+list1[[i]]%*%list2[[i]]
    }
  }
  return(a)
}

###############################################################
# Simulate from logvMEM model with known error distribution:
###############################################################
sim_logvMEM <- function(N,d,omega,A_matrix_list,B_matrix_list,beta_par,burn_in){
  # Parameters of multivariate gamma error (fixed and known):
  beta <- beta_par; alpha <- beta/2; lambda_vec <- rep(beta/2,d);theta_vec <- rep(beta,d)
  epsilon_t <- simulate_mvgamma(N+burn_in,m = d,alpha,beta,lambda_vec,theta_vec)
  log_mu_t <- matrix(NA,nrow = N+burn_in,ncol=d)
  
  p <- length(A_matrix_list)
  q <- length(B_matrix_list)
  r <- max(p,q)
  log_mu_t[1:r,] <- rep(0.1,r)
  
  x_t <- matrix(NA,nrow = N+burn_in,ncol=d)
  x_t[1:r,] = exp(log_mu_t[1:r,]) * epsilon_t[1:r,]
  log_x_t <- log(x_t)
  
  for(t in (r+1):(N+burn_in)){
    if(p>0 & q>0){
      log_mu_t[t,] <-  omega +
        list.sum.product(list1 = A_matrix_list
                         ,list2 = generate_lag_matrix(x_t = log_x_t,t=t,lag_order = r)) +
        list.sum.product(list1 = B_matrix_list
                         ,list2 = generate_lag_matrix(x_t = log_mu_t,t = t,lag_order = r))
      x_t[t,] <- exp(log_mu_t[t,])*epsilon_t[t,]
    }
    if(p>0 & q == 0){
      log_mu_t[t,] <-  omega +
        list.sum.product(list1 = A_matrix_list
                         ,list2 = generate_lag_matrix(x_t = log_x_t,t=t,lag_order = r)) 
      x_t[t,] <- exp(log_mu_t[t,])*epsilon_t[t,]
      
    }
    if(p==0 & q>0){
      log_mu_t[t,] <-  omega +
        list.sum.product(list1 = B_matrix_list
                         ,list2 = generate_lag_matrix(x_t = log_mu_t,t = t,lag_order = r))
      x_t[t,] <- exp(log_mu_t[t,])*epsilon_t[t,]
      
    }
    log_x_t <- log(x_t)
  }
  return(x_t[-c(1:burn_in),]) # Returns a N*m matrix
}

####################
# Parameters:
####################

# d <- 3 # dimension of ts
# p <- 5 # maximal lag order
# L <- matrix(c(rep(5,d),rep(2,d),rep(3,d)),nrow=d,byrow = TRUE) # component-wise Hlag structure matrix
# N <- 1000 #length of ts
# omega <- c(0.1,0.1,0.1)
# 
# A_matrix_list <- autoregressive_matrix(d = d,p = p,L = L)$A_matrix_list # A-matrices
# B_matrix_list <- NULL
# burn_in <- 1000
# 
# # Simulated data:
# data_logvMEM <- sim_logvMEM(N=N,d = d,omega = omega,A_matrix_list = A_matrix_list
#                             ,B_matrix_list = B_matrix_list,burn_in = burn_in)
# 
# # Plotting the data:
# plot.ts(data_logvMEM)