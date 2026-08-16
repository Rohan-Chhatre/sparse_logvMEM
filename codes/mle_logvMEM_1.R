
# Load libraries:
library(ACDm)
library(nloptr)
library(Rsolnp)
library(dplyr)
library(doSNOW)
project_dir = '/Users/rohan/Documents/Rohan/Finance paper/Sparse-Logarithmic-Vector-Multiplicative-Error-Model-with-Multivariate-Gamma-Errors-develop_RCodes/'
setwd(project_dir)
# Source codes:
source("codes/generate_autoregressive_matrices.R")
source("codes/sim_logvMEM.R")
source("codes/multivariate_gamma_dist.R")

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

list.multiply <- function(list1,list2){
  a <- 0
  if(length(list1) == length(list2)){
    for(i in 1:length(list1)){
      m1 <- t(list1[[i]]) #matrix of dim d*d
      m2 <- as.matrix(unlist(list2[[i]]),ncol=1) # vector of dim d
      # a <- a+crossprod(list1[[i]],list2[[i]])
      a <- a + crossprod(m1,m2)
      
    }
  }
  return(a)
}

########################################
# Log-likelihood function for logvMEM:
########################################
#log_lik_logvMEM <- function(X,A_matrix,beta_par){
#  all_param <- unlist(lapply(1:ncol(A_matrix),function(col.index) A_matrix[,col.index]))
#  N <- nrow(X)
#  d <- ncol(X)

# logvMEM parameters:
#  logvMEM_par <- all_param
#  A_matrices <- logvMEM_par[(1):(p*d^2)]
# if(q==0){
#   B_matrices <- NULL
# }else{
#   B_matrices <- logvMEM_par[(p*d^2+1):length(logvMEM_par)]
# }

#  A_matrix_list <- list()
#  start_id <- seq(1,p*d^2,by=d^2)
#  end_id <- seq(d^2,p*d^2,by=d^2)
#  A_matrix_list <- lapply(1:p,function(i) matrix(A_matrices[start_id[i]:end_id[i]],nrow = d,ncol=d))

# if(q==0){
#   B_matrix_list <- NULL
# }else{
#   B_matrix_list <- list()
#   start_id <- seq(1,q*d^2,by=d^2)
#   end_id=seq(d^2,q*d^2,by=d^2)
#   for(i in 1:q){
#     B_matrix_list[[i]] <- matrix(B_matrices[start_id[i]:end_id[i]],nrow = d,ncol=d)
#   }
#
# }

# Generate mu_t:
# log_mu_t <- matrix(NA,nrow = N,ncol=d)
#  log_mu_t[1:p,] <- rep(0.1,p) #initial values

# Writing omega in terms of the elements of matrix A and data
#  mu_vec <- apply(X,2,mean)
# variance_vec <- diag(cov(X))
#  M <- rep(digamma(beta_par),d) - log(beta_par)
#  omega <- crossprod(t((diag(x = 1,nrow = d,ncol = d) - Reduce(f = "+",x = A_matrix_list)))
#                     , matrix((log(mu_vec) - variance_vec/(2*mu_vec^2)))) - M

################
#  log_x_t <- log(X)

#  log_mu_t[(p+1):N,] <- do.call(rbind,lapply((p+1):N, function(t) t(omega +
#                                                                      list.multiply(list1 = A_matrix_list
#                                                                                    ,list2 = generate_lag_matrix(x_t = log_x_t,t=t,lag_order = p)))))
#  mu_t <- exp(log_mu_t)

# Likelihood calculation:
#  loglik1 <- -sum(log_mu_t[-c(1:p),])
#  loglik2 <- ((beta_par/2)*log(beta_par) - log(gamma(beta_par/2))) * ((N - p - 2)*d + 1)

#  a <- vector()
#  for(i in 1:d){
#    for(t in (p+1):N){
#      x <- X[t,i]
#      mu <- mu_t[t,i]
#      
#      integrand <- function(z){
#        (x/mu - z)^(beta_par/2 - 1) * exp(-beta_par*(x/mu - z)) * beta_par^(beta_par/2)/gamma(beta_par/2) * z^(beta_par/2 - 1) * exp(-beta_par*z)
#      }

#      #b_t <- min(X[t,]/mu_t[t,])
#      b_t <- x/mu
#b_t <- 100
#      a <- c(a,log(integrate(integrand,lower = 0,upper = b_t)$value))
#    }
#  }

#  loglik3 <- sum(a)

#loglik3 <- sum(data_matrix,na.rm = T)
#  loglik <- sum(loglik1,loglik2,loglik3)
#  return(loglik)
#}

########################################
# Full log-likelihood for logvMEM
# using CORRECTED MVGamma density
########################################
log_lik_logvMEM <- function(X, A_matrix, beta_par) {
  N <- nrow(X)
  d <- ncol(X)
  
  all_param <- unlist(lapply(1:ncol(A_matrix), function(col.index) A_matrix[, col.index]))
  A_matrices <- all_param[1:(p * d^2)]
  
  start_id <- seq(1, p * d^2, by = d^2)
  end_id   <- seq(d^2, p * d^2, by = d^2)
  A_matrix_list <- lapply(1:p, function(i) {
    matrix(A_matrices[start_id[i]:end_id[i]], nrow = d, ncol = d)
  })
  
  # conditional means
  log_mu_t <- matrix(NA_real_, nrow = N, ncol = d)
  log_mu_t[1:p, ] <- rep(0.1, p)
  
  mu_vec <- colMeans(X)
  variance_vec <- diag(cov(X))
  M <- rep(digamma(beta_par), d) - log(beta_par)
  
  omega <- crossprod(
    t(diag(1, d) - Reduce(`+`, A_matrix_list)),
    matrix(log(mu_vec) - variance_vec / (2 * mu_vec^2))
  ) - M
  
  log_x_t <- log(X)
  
  log_mu_t[(p + 1):N, ] <- do.call(rbind, lapply((p + 1):N, function(tt) {
    t(omega + list.multiply(
      list1 = A_matrix_list,
      list2 = generate_lag_matrix(x_t = log_x_t, t = tt, lag_order = p)
    ))
  }))
  
  mu_t <- exp(log_mu_t)
  
  # symmetric parameterization tied to beta_par
  alpha <- beta_par / 2
  beta  <- beta_par
  lambda_vec <- rep(beta_par / 2, d)
  theta_vec  <- rep(beta_par, d)
  
  ll <- 0
  for (tt in (p + 1):N) {
    eps_t <- X[tt, ] / mu_t[tt, ]
    
    logdens <- mvgamma_pdf(
      x_vec = eps_t,
      alpha = alpha,
      beta = beta,
      lambda_vec = lambda_vec,
      theta_vec = theta_vec,
      log = TRUE
    )
    # for which Y-t > LARGEST NO. Fetch t.
    if (!is.finite(logdens)) return(-Inf)
    
    ll <- ll + logdens - sum(log_mu_t[tt, ])
  }
  
  ll
}

########################################################
#This step is wrong because of the OLD MVGamma likelihood.
########################################################
#log_lik_logvMEM_ith_component <- function(X,A_matrix_i,i.index,beta_par){
#  N <- nrow(X)
#  d <- ncol(X)
#  all_param_i <- A_matrix_i
# # mvgamma parameters (fixed and known):
# mvgamma_par <- c(alpha=beta_par/2,beta=beta_par,lambda_vec=rep(beta_par/2,d),theta_vec=rep(beta_par,d))
# alpha <- mvgamma_par[1]
# beta <- mvgamma_par[2]
# lambda_vec <- mvgamma_par[3:(2+d)]
# theta_vec <- mvgamma_par[(2+d+1):(2+2*d)]

# logvMEM parameters:
#  logvMEM_par <- all_param_i
#  A_matrices <- logvMEM_par[1:(p*d)]
# omega <- logvMEM_par[1]
# if(p == 0){
#   A_matrices <- NULL
# }else{
#   A_matrices <- logvMEM_par[1:(p*d)]
# }
# if(q==0){
#   B_matrices <- NULL
# }else{
#   B_matrices <- logvMEM_par[(p*d+1):length(logvMEM_par)]
# }

# Writing omega in terms of the elements of matrix A and data
#  start_id <- seq(1,length(A_matrices),by=d)
#  end_id <- seq(d,length(A_matrices),by=d)
#  A_matrices_list <- list()
#  for(i in 1:length(start_id)){
#    A_matrices_list[[i]] <- A_matrices[start_id[i]:end_id[i]]
#  }
#  mu_vec <- colMeans(X)
#  variance_vec <- diag(cov(X))
#  M <- rep(digamma(beta_par)- log(beta_par),d)[i.index] 
#  omega <- tcrossprod((diag(x = 1,nrow = d)[i.index,] - Reduce(f = "+",x = A_matrices_list)) ,  t(matrix((log(mu_vec) - variance_vec/(2*mu_vec^2))))) - M

#  log_x_t <- log(X)
#  log_mu_i <- rep(0,N)
#  for(t in (p+1):N){
#    log_mu_i[t] <- omega + crossprod(A_matrices , matrix(unlist(generate_lag_matrix(x_t = log_x_t,t=t,lag_order = p))))
#  }

#  mu_i <- exp(log_mu_i)

# Likelihood calculation based on ith component:
#  loglik1 <- -sum(log_mu_i[-c(1:p)])
#  loglik2 <- (((beta_par/2)*log(beta_par) - log(gamma(beta_par/2))) * ((N - p - 2)*d + 1))/d

#  a <- vector()
#  for(i in 1:d){
#    for(t in (p+1):N){
#      x <- X[t,i.index]
#      mu <- mu_i[t]

#      integrand <- function(z){
#        (x/mu - z)^(beta_par/2 - 1) * exp(-beta_par*(x/mu - z)) * beta_par^(beta_par/2)/gamma(beta_par/2) * z^(beta_par/2 - 1) * exp(-beta_par*z)
#      }

#      b_t <- x/mu
#      #b_t <- x/mu
#      #b_t <- 100
#      a <- c(a,log(integrate(integrand,lower = 0,upper = b_t)$value))
#    }
#  }

#  loglik3 <- sum(a)
#  loglik <- sum(loglik1,loglik2,loglik3)
#  return(loglik)
#}
#########################################################
# Componentwise objective using full corrected likelihood
# Only row i.index is optimized, others are fixed
#########################################################
#log_lik_logvMEM_ith_component <- function(X, A_matrix_i, i.index, beta_par, A_matrix_fixed) {
#  A_full <- as.matrix(A_matrix_fixed)
#  A_full[i.index, ] <- A_matrix_i
#  log_lik_logvMEM(X = X, A_matrix = A_full, beta_par = beta_par)
#}
#########################################
# Function to calculate initial values: using an ACD Model
#########################################
init.val <- function(X,maxlag,n_beta){
  p <- maxlag
  d <- ncol(X)
  
  # All initial values:
  init_val_list <- list()
  par_matrix <- matrix(NA,nrow=d,ncol=p)
  par_matrix_list <- list()
  beta_i <- c()
  
  for(i in 1:d){
    lacd_fit <- acdFit(durations = X[,i], model = "LACD1",
                       dist = "gengamma", order = c(p,1),forceErrExpec = TRUE
                       ,startPara = c(0.1,rep(0.1,p),0,3,1),fixedParamPos = c(FALSE,rep(FALSE,p),TRUE,FALSE,TRUE))
    
    beta_i[i] <- lacd_fit$parameterInference["kappa",1]
    for(lag_id in 1:p){
      par_matrix[i,lag_id] <- lacd_fit$parameterInference[paste0("alpha",lag_id),1]
      par_matrix_list[[lag_id]] <- diag(par_matrix[,lag_id])
    }
  }
  
  beta_vec <- seq(ifelse(min(beta_i)-2>0,min(beta_i)-2,1),ifelse(max(beta_i)+2>0,max(beta_i+2),10)
                  ,length.out=n_beta)
  beta_vec_new <- seq(ifelse(min(beta_i)>0,min(beta_i),1),ifelse(max(beta_i)>0,max(beta_i),10)
                      ,length.out=n_beta)
  
  return(list(
    init_A_matrix = do.call(cbind, par_matrix_list),
    beta_list = beta_vec,
    beta_list_max_min = beta_vec_new,
    beta_init = median(beta_i),
    beta_i = beta_i))
}

#################################
# Profile Likelihood estimation:
#################################

#mle_ith_comp <- function(X,i.index,maxlag,init_A_matrix,beta_par,max_iter_bobyqa=3000,solver,eps=1e-3){
#  p <- maxlag
#  d <- ncol(X)

# Bounds and max iterations for the parameters:
#  lb <- rep(-0.5,p*d)
#  ub <- rep(1,p*d)

#  if(solver == "bobyqa"){  
#    # Using bobyqa from nloptr:
#    all_param_list <- tryCatch({bobyqa(x0 = init_A_matrix[i.index,]
#                                       ,fn = function(param_i) -log_lik_logvMEM_ith_component(X=X,A_matrix_i = param_i
#                                                                                              ,i.index = i.index,beta_par = beta_par),
#                                       lower = lb,upper = ub, control = list(maxeval = max_iter_bobyqa,xtol_rel=eps))},
#                               error=function(e) {
#                                return(NA)
#                               })
# Parameter estimates for each component:
#    A_i_estimate <- all_param_list$par
# Convergence for each component:
#    conv_i <- all_param_list$convergence
# Iteration for each component:
#    iter_i <- all_param_list$iter
# Log-likelihood for each component:
#    loglik_i <- log_lik_logvMEM_ith_component(X = X,A_matrix_i = A_i_estimate
#                                              ,beta_par = beta_par,i.index = i.index)

#  }

#  if(solver == "solnp"){
# Using solnp from package Rsolnp:
#    all_param_list <- tryCatch({solnp(pars = init_A_matrix[i.index,]
#                                     ,fun = function(param_i) -log_lik_logvMEM_ith_component(X=X,A_matrix_i = param_i
#                                                                                              ,i.index = i.index,beta_par = beta_par),
#                                      LB = lb,UB = ub, control = list(tol=eps,trace=1))},
#                               error=function(e) {
#                                 return(NA)
#                               }) 
#    
#    # Parameter estimates for each component:
#    A_i_estimate <- all_param_list$pars
#    # Convergence for each component:
#    conv_i <- all_param_list$convergence
#    # Iteration for each component:
#    iter_i <- all_param_list$nfuneval
#    # Log-likelihood for each component:
#    loglik_i <- log_lik_logvMEM_ith_component(X = X,A_matrix_i = A_i_estimate
#                                              ,beta_par = beta_par,i.index = i.index)
#    
#  }
#  return(list(A_i_estimate = A_i_estimate,conv_i = conv_i,iter_i = iter_i
#              ,loglik_i = loglik_i))
#}

#################################
# Profile likelihood estimation:
# componentwise block update
#################################
#mle_ith_comp <- function(X, i.index, maxlag, init_A_matrix, beta_par,
#                         max_iter_bobyqa = 3000, solver, eps = 1e-3) {
#  p <- maxlag
#  d <- ncol(X)
#  
#  lb <- rep(-0.5, p * d)
#  ub <- rep(1, p * d)
#  
#  A_fixed <- as.matrix(init_A_matrix)
#  
#  if (solver == "bobyqa") {
#    all_param_list <- tryCatch({
#      bobyqa(
#        x0 = A_fixed[i.index, ],
#        fn = function(param_i) -log_lik_logvMEM_ith_component(
#         X = X,
#          A_matrix_i = param_i,
#          i.index = i.index,
#          beta_par = beta_par,
#          A_matrix_fixed = A_fixed
#        ),
#        lower = lb,
#        upper = ub,
#        control = list(maxeval = max_iter_bobyqa, xtol_rel = eps)
#      )
#    }, error = function(e) NA)
#    
#    if (length(all_param_list) == 1 && is.na(all_param_list)) return(NA)
#    
#    A_i_estimate <- all_param_list$par
#    conv_i <- all_param_list$convergence
#    iter_i <- all_param_list$iter
#    
# evaluate full likelihood at updated block
#    A_fixed[i.index, ] <- A_i_estimate
#    loglik_i <- log_lik_logvMEM(X = X, A_matrix = A_fixed, beta_par = beta_par)
#  }

#  if (solver == "solnp") {
#    all_param_list <- tryCatch({
#      solnp(
#        pars = A_fixed[i.index, ],
#        fun = function(param_i) -log_lik_logvMEM_ith_component(
#          X = X,
#          A_matrix_i = param_i,
#          i.index = i.index,
#          beta_par = beta_par,
#          A_matrix_fixed = A_fixed
#        ),
#        LB = lb,
#        UB = ub,
#        control = list(tol = eps, trace = 1)
#      )
#    }, error = function(e) NA)

#    if (length(all_param_list) == 1 && is.na(all_param_list)) return(NA)
#    
#    A_i_estimate <- all_param_list$pars
#    conv_i <- all_param_list$convergence
#    iter_i <- all_param_list$nfuneval
#    
#    A_fixed[i.index, ] <- A_i_estimate
#    loglik_i <- log_lik_logvMEM(X = X, A_matrix = A_fixed, beta_par = beta_par)
#  }

#  list(
#    A_i_estimate = A_i_estimate,
#    conv_i = conv_i,
#    iter_i = iter_i,
#    loglik_i = loglik_i
#  )
#}

#R-Matlab optimizations: COMMENTS FROM NR

############################################################
# Method 1: Alternating joint estimation of (A, beta)
# - A updated blockwise by row
# - beta updated by 1D optimization
############################################################

library(nloptr)

############################################################
# Update one row/block of A while holding others fixed
############################################################
############################################################
#Glossary
#X = data 
#A_current = autoregressive matrix


############################################################
update_A_block <- function(X, A_current, i.index, beta_par,
                           lower = -0.5, upper = 1,
                           max_iter_bobyqa = 1000, eps = 1e-4) {
  d <- ncol(X)
  p <- ncol(A_current) / d
  
  stopifnot(nrow(A_current) == d)
  stopifnot(ncol(A_current) == p * d)
  
  lb <- rep(lower, p * d)
  ub <- rep(upper, p * d)
  
  obj_fun <- function(a_i_vec) {
    A_trial <- A_current
    A_trial[i.index, ] <- a_i_vec
    val <- -log_lik_logvMEM(X = X, A_matrix = A_trial, beta_par = beta_par)
    
    if (!is.finite(val)) val <- 1e12
    val
  }
  #BOBYQA performs derivative-free bound-constrained optimization using an iteratively constructed quadratic approximation for the objective function.
  fit <- tryCatch(
    bobyqa(
      x0 = as.numeric(A_current[i.index, ]),
      fn = obj_fun,
      lower = lb,
      upper = ub,
      control = list(maxeval = max_iter_bobyqa, xtol_rel = eps)
    ),
    error = function(e) NULL
  )
  
  if (is.null(fit) || is.null(fit$par) || any(!is.finite(fit$par))) {
    return(list(
      A_i_estimate = as.numeric(A_current[i.index, ]),
      convergence = NA,
      iter = NA,
      objective = obj_fun(as.numeric(A_current[i.index, ]))
    ))
  }
  
  list(
    A_i_estimate = fit$par,
    convergence = fit$convergence,
    iter = fit$iter,
    objective = fit$value
  )
}

############################################################
# One full sweep over all rows of A
############################################################
update_A_blockwise <- function(X, A_init, beta_par,
                               max_block_sweeps = 1,
                               lower = -0.5, upper = 1,
                               max_iter_bobyqa = 1000, eps = 1e-4,
                               trace = TRUE) {
  A_current <- A_init
  d <- ncol(X)
  
  block_log <- vector("list", max_block_sweeps)
  
  for (sweep in seq_len(max_block_sweeps)) {
    if (trace) cat("  A-block sweep:", sweep, "\n")
    
    row_log <- vector("list", d)
    
    for (i in seq_len(d)) {
      upd <- update_A_block(
        X = X,
        A_current = A_current,
        i.index = i,
        beta_par = beta_par,
        lower = lower,
        upper = upper,
        max_iter_bobyqa = max_iter_bobyqa,
        eps = eps
      )
      
      A_current[i, ] <- upd$A_i_estimate
      row_log[[i]] <- upd
    }
    
    ll_now <- log_lik_logvMEM(X = X, A_matrix = A_current, beta_par = beta_par)
    
    block_log[[sweep]] <- list(
      sweep = sweep,
      loglik = ll_now,
      row_updates = row_log
    )
    
    if (trace) cat("    loglik after sweep =", ll_now, "\n")
  }
  
  list(
    A_estimate = A_current,
    loglik = log_lik_logvMEM(X = X, A_matrix = A_current, beta_par = beta_par),
    block_log = block_log
  )
}

############################################################
# Update beta while holding A fixed
# 1D optimization
############################################################
update_beta_1d <- function(X, A_matrix,
                           beta_lower = 0.2,
                           beta_upper = 20,
                           tol = 1e-3) {
  obj_fun <- function(beta_par) {
    if (!is.finite(beta_par) || beta_par <= 0) return(1e12)
    
    val <- -log_lik_logvMEM(X = X, A_matrix = A_matrix, beta_par = beta_par)
    if (!is.finite(val)) val <- 1e12
    val
  }
  #The function optimize searches the interval from lower to upper for a minimum or maximum of the function f with respect to its first argument.
  fit <- tryCatch(
    optimize(f = obj_fun, interval = c(beta_lower, beta_upper), tol = tol),
    error = function(e) NULL
  )
  
  if (is.null(fit)) {
    beta_fallback <- (beta_lower + beta_upper) / 2
    return(list(
      beta_estimate = beta_fallback,
      objective = obj_fun(beta_fallback)
    ))
  }
  
  list(
    beta_estimate = fit$minimum,
    objective = fit$objective
  )
}

############################################################
# Main Method 1 estimator
# Alternating optimization of A and beta
############################################################
estimate_method1_joint <- function(X,
                                   init_A_matrix,
                                   init_beta,
                                   max_outer_iter = 10,
                                   max_block_sweeps = 1,
                                   lower_A = -0.5,
                                   upper_A = 1,
                                   beta_lower = 0.2,
                                   beta_upper = 20,
                                   max_iter_bobyqa = 1000,
                                   eps_A = 1e-4,
                                   tol_beta = 1e-3,
                                   tol_outer_A = 1e-6,
                                   tol_outer_beta = 1e-6,
                                   trace = TRUE) {
  A_current <- init_A_matrix
  beta_current <- init_beta
  
  if (!is.matrix(A_current)) A_current <- as.matrix(A_current)
  if (!is.numeric(beta_current) || length(beta_current) != 1 || beta_current <= 0) {
    stop("init_beta must be a positive scalar.")
  }
  
  history <- vector("list", max_outer_iter)
  
  ll_current <- log_lik_logvMEM(
    X = X,
    A_matrix = A_current,
    beta_par = beta_current
  )
  
  if (trace) {
    cat("Initial beta =", beta_current, "\n")
    cat("Initial loglik =", ll_current, "\n")
  }
  
  for (iter in seq_len(max_outer_iter)) {
    if (trace) cat("\nOuter iteration:", iter, "\n")
    
    A_old <- A_current
    beta_old <- beta_current
    
    ##################################################
    # Step 1: update A blockwise given beta
    ##################################################
    
    ll_before_A <- ll_current
    
    A_upd <- update_A_blockwise(
      X = X,
      A_init = A_current,
      beta_par = beta_current,
      max_block_sweeps = max_block_sweeps,
      lower = lower_A,
      upper = upper_A,
      max_iter_bobyqa = max_iter_bobyqa,
      eps = eps_A,
      trace = trace
    )
    
    A_current <- A_upd$A_estimate
    
    ll_after_A <- log_lik_logvMEM(
      X = X,
      A_matrix = A_current,
      beta_par = beta_current
    )
    
    delta_ll_A_rel <- abs(ll_after_A - ll_before_A) /
      max(1, abs(ll_before_A))
    
    ##################################################
    # Step 2: update beta given updated A
    ##################################################
    
    beta_upd <- update_beta_1d(
      X = X,
      A_matrix = A_current,
      beta_lower = beta_lower,
      beta_upper = beta_upper,
      tol = tol_beta
    )
    
    beta_current <- beta_upd$beta_estimate
    
    ll_new <- log_lik_logvMEM(
      X = X,
      A_matrix = A_current,
      beta_par = beta_current
    )
    
    delta_ll_beta_rel <- abs(ll_new - ll_after_A) /
      max(1, abs(ll_after_A))
    
    ##################################################
    # Diagnostics: keep old parameter changes too
    ##################################################
    
    delta_A <- sqrt(sum((A_current - A_old)^2))
    delta_beta <- abs(beta_current - beta_old)
    
    history[[iter]] <- list(
      iter = iter,
      A = A_current,
      beta = beta_current,
      loglik = ll_new,
      delta_A = delta_A,
      delta_beta = delta_beta,
      delta_ll_A_rel = delta_ll_A_rel,
      delta_ll_beta_rel = delta_ll_beta_rel,
      A_update_log = A_upd$block_log,
      beta_objective = beta_upd$objective
    )
    
    if (trace) {
      cat("  Updated beta =", beta_current, "\n")
      cat("  Updated loglik =", ll_new, "\n")
      cat("  delta_A =", delta_A, "\n")
      cat("  delta_beta =", delta_beta, "\n")
      cat("  delta_ll_A_rel =", delta_ll_A_rel, "\n")
      cat("  delta_ll_beta_rel =", delta_ll_beta_rel, "\n")
    }
    
    ##################################################
    # Stopping rule: relative likelihood change
    ##################################################
    
    if (delta_ll_A_rel < tol_outer_A &&
        delta_ll_beta_rel < tol_outer_beta) {
      
      if (trace) {
        cat("Converged based on relative log-likelihood change.\n")
      }
      
      history <- history[seq_len(iter)]
      
      return(list(
        A_estimate = A_current,
        beta_estimate = beta_current,
        loglik = ll_new,
        n_outer_iter = iter,
        converged = TRUE,
        history = history
      ))
    }
    
    ll_current <- ll_new
  }
  
  list(
    A_estimate = A_current,
    beta_estimate = beta_current,
    loglik = ll_current,
    n_outer_iter = max_outer_iter,
    converged = FALSE,
    history = history
  )
}

############################################################
# Method 2: Blockwise profile likelihood estimation
# - beta is profiled over a grid
# - for each fixed beta, A is estimated blockwise
############################################################




#####################Testing####################################


# ################################
# # Simulate data from logvMEM:
# ################################
# d <- 3
# p <- 3
# q <- 0
# r <- max(p,q)
# L <- matrix(c(rep(2,d),rep(3,d),rep(3,d)),nrow=d,ncol=d,byrow = TRUE) # HLag- componentwise
# A_matrix_list <- autoregressive_matrix(d = d,p = p,L = L,max_eigen_abs = 0.95)$A_matrix_list
# 
# X <- sim_logvMEM(N = 1500,d = d,omega=matrix(rep(0.1,d),ncol=1),A=A_matrix_list
#                  ,B=NULL,beta_par=4,burn_in = 500)
# A_matrix <- do.call(cbind,A_matrix_list)
# 
# # Plotting the time series and autocorrelation:
# plot.ts(X)
# cor(X,method = "kendall")

##################################################################################################
# Example to test whether the sum of individual lpglik is equal to the loglik of the full model:
##################################################################################################
#for(beta_par in 1:8){
#  l1= log_lik_logvMEM(X,A_matrix,beta_par)
#  print(l1)
#  
#  l2 = sum(unlist(lapply(1:d,function(i.index) log_lik_logvMEM_ith_component(X,A_matrix_i=A_matrix[i.index,],i.index=i.index,beta_par))))
#  print(l2)
#}

# # Testing:
#all_init_values <- init.val(X,maxlag=p,n_beta=10)
#init_A_matrix <- all_init_values$init_A_matrix
#beta_grid <- all_init_values$beta_list

# Testing:
#mle_i_output <- mle_ith_comp(X = X,i.index = 2,maxlag = p,init_A_matrix = all_init_values$init_A_matrix
#                             ,max_iter_bobyqa = 3000,solver = "bobyqa",eps = 1e-3,beta_par = 4)

#Profile likelihood estimation:
#beta_i_index <- expand.grid(i_index = 1:d,beta = beta_grid)
#beta_grid_df <- data.frame(beta=beta_grid,model_id=1:length(beta_grid))
#beta_i_index_df <- inner_join(beta_i_index,beta_grid_df)

#library(doParallel)

# Set up parallel estimation:
#no_cores <- detectCores()
#cl <- makeCluster(no_cores - 1)
#registerDoParallel(cl)
#registerDoSNOW(cl)

# print out the progress for every iteration
#progress <- function(n) cat(sprintf("task %d is complete\n", n))
#opts <- list(progress=progress)
# 
# 
#X_train = X
# 
#mle_output_list <- foreach(i=1:nrow(beta_i_index_df), .packages=c("ACDm","nloptr","Rsolnp"),
#                           .export = ls(globalenv()),.verbose = TRUE,.options.snow = opts) %dopar% {
#                             tryCatch({mle_ith_comp(X=X_train,i.index = beta_i_index_df$i_index[i],maxlag=p
#                                                    ,init_A_matrix = init_A_matrix,beta_par=beta_i_index_df$beta[i]
#                                                    ,max_iter_bobyqa = 3000,solver="bobyqa",eps=1e-3)}
#                                      ,error=function(e){return(NA)})
#                           }
# 
#stopCluster(cl)
# 
# # Binding all outputs:
#all_mle_list <- list()
#for(i in 1:length(mle_output_list)){
#  mle_list <- mle_output_list[[i]]
#  if(is.na(mle_list[1])==T){
#    all_mle_list[[i]] <- NA
#  }else{
#    mle_df = cbind.data.frame(beta_i_index_df[i,],t(mle_list$A_i_estimate),convergence = mle_list$conv_i
#                              ,n_iterations = mle_list$iter_i,loglik_i = mle_list$loglik_i)
#    colnames(mle_df) <- c(colnames(beta_i_index_df),
#                          paste("A",1:(p*d),sep="_"),"convergence","n_iterations"
#                          ,"loglik_i")
#    all_mle_list[[i]] <- mle_df
#  }
#}

# # Row binding all mle results as data frame:
#all_mle_df <- do.call(rbind,all_mle_list)
# 
# # Take only the complete cases:
#all_mle_df <- all_mle_df[complete.cases(all_mle_df),]
# 
# # Filter out the models with incomplete information:
#split_all_mle_df_bymodelid <- split.data.frame(all_mle_df,f = all_mle_df$model_id)
#rm_id <- which(lapply(split_all_mle_df_bymodelid,nrow)<d)
#if(length(rm_id)>0){
#  split_all_mle_df_bymodelid <- split_all_mle_df_bymodelid[-rm_id]
#}
# 
# # Adding full data log-lik in the data frame:
#for(i in 1:length(split_all_mle_df_bymodelid)){
#  split_all_mle_df_bymodelid[[i]]$loglik_full_data <- sum(split_all_mle_df_bymodelid[[i]]$loglik_i)
#}
# 
# # Calculating log-likelihood by model_id:
#mle_id <- which.max(unlist(lapply(1:length(split_all_mle_df_bymodelid),function(i) unique(split_all_mle_df_bymodelid[[i]]$loglik_full_data))))
#A_matrix_estimated_mle <- split_all_mle_df_bymodelid[[mle_id]][,paste("A",1:(p*d),sep="_")]
#beta_mle <- unique(split_all_mle_df_bymodelid[[mle_id]][,"beta"])
