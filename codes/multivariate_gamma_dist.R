
#############################################
# Function to calculate the pdf of mvgamma:
#############################################
#mvgamma_pdf <- function(x_vec,alpha,beta,lambda_vec,theta_vec){
#  m <- length(x_vec)
  
  # Constant:
#  const <- (beta^alpha)/gamma(alpha)
#  B <- min(x_vec)
  
  # Integrand function:
#  integrand <- function(z){
#    val <- (x - z)^(lambda - 1)* exp(-theta_vec[i]*(x - z)) * z^(alpha - 1) * exp(-beta*z)
#  }
  
#  prod_val <- rep(NA,m)
#  for(i in 1:m){
#    x <- x_vec[i] 
#    lambda <- lambda_vec[i]
#    integrand_val <- integrate(integrand,lower = 0,upper = B)
#    prod_val[i] <- theta_vec[i]^(lambda_vec[i])/gamma(lambda_vec[i]) * integrand_val$value
#  }
#  
#  # Final value of pdf:
#  pdf_val <- prod(prod_val)*const
#  return(pdf_val)
#}
#############################################
# Corrected multivariate gamma pdf
# Based on:
# X_i = V_i + Z
# V_i ~ Gamma(lambda_i, rate = theta_i)
# Z   ~ Gamma(alpha, rate = beta)
#############################################

mvgamma_pdf <- function(x_vec, alpha, beta, lambda_vec, theta_vec, log = FALSE) {
  x_vec <- as.numeric(x_vec)
  lambda_vec <- as.numeric(lambda_vec)
  theta_vec <- as.numeric(theta_vec)
  
  m <- length(x_vec)
  
  if (length(lambda_vec) != m) stop("length(lambda_vec) must match length(x_vec)")
  if (length(theta_vec) != m) stop("length(theta_vec) must match length(x_vec)")
  
  if (any(!is.finite(x_vec)) || any(!is.finite(lambda_vec)) || any(!is.finite(theta_vec))) {
    return(if (log) -Inf else 0)
  }
  
  if (alpha <= 0 || beta <= 0 || any(lambda_vec <= 0) || any(theta_vec <= 0)) {
    return(if (log) -Inf else 0)
  }
  
  if (any(x_vec <= 0)) {
    return(if (log) -Inf else 0)
  }
  
  B <- min(x_vec)
  
  if (B <= 0) {
    return(if (log) -Inf else 0)
  }
  
  # log constant outside the integral
  log_const <- alpha * log(beta) - lgamma(alpha) +
    sum(lambda_vec * log(theta_vec) - lgamma(lambda_vec))
  
  # joint integrand on log scale
  log_integrand <- function(z) {
    if (z <= 0 || z >= B) return(-Inf)
    
    shifted <- x_vec - z
    if (any(shifted <= 0)) return(-Inf)
    
    sum((lambda_vec - 1) * log(shifted) - theta_vec * shifted) +
      (alpha - 1) * log(z) - beta * z
  }
  
  integrand <- function(z) {
    exp(vapply(z, log_integrand, numeric(1)))
  }
  
  int_val <- tryCatch(
    integrate(integrand, lower = 0, upper = B,
              subdivisions = 200L, rel.tol = 1e-3, stop.on.error = FALSE)$value,
    error = function(e) NA_real_
  )
  
  if (!is.finite(int_val) || is.na(int_val) || int_val <= 0) {
    return(if (log) -Inf else 0)
  }
  
  log_pdf <- log_const + log(int_val)
  
  if (log) return(log_pdf)
  exp(log_pdf)
}

#######
#Testing
#mvgamma_pdf = 1
#######
#integrate using hcubature


# #################
# # Parameters:
# #################
# m <- 5 #dimension
# alpha <- 2
# beta <- 6
# lambda_vec <- c(1,3,1,1,1)
# theta_vec <- c(1,1,1,1,1)
 
# x_vec <- c(1,1,1,1,4)
 
 # Evaluation of pdf:
# mvgamma_pdf(x_vec,alpha,beta,lambda_vec,theta_vec,log = FALSE)

#########################################################################
# Simulate N observations from a m-dimensional multivariate Gamma vector
# of observations (Tsionas, 2003):
#################################################################

simulate_mvgamma <- function(N,m,alpha,beta,lambda_vec,theta_vec){
  z <- rgamma(n=N,shape = alpha,scale = 1/beta)
  v_mat <- matrix(NA,nrow=N,ncol = m)
  x_mat <- matrix(NA,nrow=N,ncol = m)
  for(i in 1:m){
    v_mat[,i] <- rgamma(n=N,shape =lambda_vec[i] ,scale =1/theta_vec[i])
    x_mat[,i] <- v_mat[,i] + z
  }
  return(x_mat)
}
  
# N <- 500
# alpha <- 1
# beta <- 2
# m <- 2
# lambda_vec <- rep(1,m)
# theta_vec <- rep(2,m)
# 
# x_mat <- simulate_mvgamma(N,m,alpha,beta,lambda_vec,theta_vec)
# 
# #######################
# # Correctness:
# #######################
# # Theoretical Mean:
# lambda_vec/theta_vec + alpha/beta
# 
# # Sample Mean:
# apply(x_mat,2,mean)
# 
# # Theoretical Covariance:
# cov_mat <- matrix(NA,nrow=m,ncol=m)
# for(i in 1:nrow(cov_mat)){
#   for(j in 1:ncol(cov_mat)){
#     if(i==j){
#       cov_mat[i,i] <- lambda_vec[i]/theta_vec[i]^2 + alpha/beta^2
#     }else{
#       cov_mat[i,j] <- alpha/beta^2
#     }
#   }
# }
# 
# cov_mat_theoretical <- cov_mat
# 
# #Sample Covariance:
# cov(x_mat)
# 




