

##########################
# Simulated Data:
##########################

#set.seed(1234)

# Source codes:
# 1. Simulation:
#source("C:/Users/chira/Documents/GitHub/Project-2/core_files/sim_logvMEM.R")
# 2. Log-likelihood:
#source("C:/Users/chira/Documents/GitHub/Project-2/core_files/log_lik_vMEM_without_omega.R")
# 3. Generate Autoregressive matrices:  
#source("C:/Users/chira/Documents/GitHub/Project-2/core_files/generate_autoregressive_matrices.R")

#set.seed(1234)
# d <- 2
# p <- 2
# q <- 0
# r <- max(p,q)
# L <- matrix(c(rep(1,2),rep(1,2)),nrow=d,ncol=d,byrow = TRUE) # HLag- componentwise
# A_matrix_list <- autoregressive_matrix(d = 2,p = 2,L = L,max_eigen_abs = 0.8)$A_matrix_list
# 
# X <- sim_logvMEM(N = 1500,d = 2,omega=matrix(c(0.1,0.1),ncol=1),A=A_matrix_list
#                  ,B=NULL,burn_in = 500)
# 
# plot.ts(X)


# i.index <- 1
# A_matrix_i <- unlist(lapply(A_matrix_list, function(x) x[i.index,]))

fitted_values_logvMEM_i <- function(X,i.index,A_matrix_i,beta_par,p,q){
  N <- nrow(X)
  d <- ncol(X)
  r <- max(p,q)
  
  # mvgamma parameters (fixed and known):
  mvgamma_par <- c(alpha=beta_par/2,beta=beta_par
                   ,lambda_vec=rep(beta_par/2,d),theta_vec=rep(beta_par,d))
  alpha <- mvgamma_par[1]
  beta <- mvgamma_par[2]
  lambda_vec <- mvgamma_par[3:(2+d)]
  theta_vec <- mvgamma_par[(2+d+1):(2+2*d)]
  
  # logvMEM parameters:
  logvMEM_par <- A_matrix_i
  # omega <- logvMEM_par[1]
  if(p == 0){
    A_matrices <- NULL
  }else{
    A_matrices <- logvMEM_par[1:(p*d)]
  }
  if(q==0){
    B_matrices <- NULL
  }else{
    B_matrices <- logvMEM_par[(p*d+1):length(logvMEM_par)]
  }
  
  # Writing omega in terms of the elements of matrix A and data
  start_id <- seq(1,length(A_matrices),by=d)
  end_id <- seq(d,length(A_matrices),by=d)
  A_matrices_list <- list()
  for(i in 1:length(start_id)){
    A_matrices_list[[i]] <- A_matrices[start_id[i]:end_id[i]]
  }
  mu_vec <- colMeans(X)
  variance_vec <- diag(cov(X))
  M <- rep(digamma(beta_par)- log(beta_par),d)[i.index] 
  omega <- (diag(x = 1,nrow = d)[i.index,] - Reduce(f = "+",x = A_matrices_list)) %*% matrix((log(mu_vec) - variance_vec/(2*mu_vec^2))) - M

  log_x_t <- log(X)
  log_mu_i <- rep(NA,N)
  log_mu_i[1:r] <- rep(0.1,r)
  for(t in (r+1):N){
    log_mu_i[t] <- omega + t(A_matrices) %*% matrix(unlist(generate_lag_matrix(x_t = log_x_t,t=t,lag_order = r)))
  }
  # Fitted values:
  fitted_values <- exp(log_mu_i)
  return(fitted_values)
}



# i.index <- 1
# A_matrix_i <- unlist(lapply(A_matrix_list, function(x) x[i.index,]))
# 
# fitted_values <- fitted_values_logvMEM(X = X,i.index = i.index,A_matrix_i = A_matrix_i,p = 2,q=0)
# 
# 
# plot.ts(X[,i.index])
# lines(fitted_values,col="red")
# 
# 
# head(X[,i.index])
# head(fitted_values)
