
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
source("codes/gpu/multivariate_gamma_dist_gpu_test.R")

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
logvMEM_unpack_A <- function(A_matrix, d = nrow(A_matrix)) {
  A_matrix <- as.matrix(A_matrix)
  if (ncol(A_matrix) %% d != 0) {
    stop("ncol(A_matrix) must be a multiple of d.")
  }
  p_local <- ncol(A_matrix) %/% d
  A_list <- lapply(seq_len(p_local), function(lag_id) {
    cols <- ((lag_id - 1L) * d + 1L):(lag_id * d)
    A_matrix[, cols, drop = FALSE]
  })
  list(p = p_local, A_list = A_list)
}

############################################################
# Vectorized conditional log-means.
# This is mathematically the same log-vMEM recursion for q = 0,
# but forms the complete lag design matrix once and uses BLAS.
############################################################
compute_log_mu_logvMEM_vectorized <- function(X, A_matrix, beta_par) {
  X <- as.matrix(X)
  A_matrix <- as.matrix(A_matrix)
  N <- nrow(X)
  d <- ncol(X)
  
  unpacked <- logvMEM_unpack_A(A_matrix, d = d)
  p_local <- unpacked$p
  A_matrix_list <- unpacked$A_list
  
  if (N <= p_local) stop("Need N > p.")
  
  mu_vec <- colMeans(X)
  variance_vec <- diag(cov(X))
  M <- rep(digamma(beta_par), d) - log(beta_par)
  
  omega <- crossprod(
    t(diag(1, d) - Reduce(`+`, A_matrix_list)),
    matrix(log(mu_vec) - variance_vec / (2 * mu_vec^2))
  ) - M
  omega <- as.numeric(omega)
  
  log_x <- log(X)
  lag_blocks <- lapply(seq_len(p_local), function(lag_id) {
    log_x[(p_local + 1L - lag_id):(N - lag_id), , drop = FALSE]
  })
  lag_design <- do.call(cbind, lag_blocks)
  
  log_mu_use <- sweep(
    lag_design %*% t(A_matrix),
    MARGIN = 2,
    STATS = omega,
    FUN = "+"
  )
  
  list(
    log_mu = log_mu_use,
    mu = exp(log_mu_use),
    omega = omega,
    p = p_local
  )
}
# ============================================================
# CUDA VERSION OF compute_log_mu_logvMEM_vectorized()
# ============================================================

# compute_log_mu_logvMEM_cuda <- function(
    #     X,
#     A_matrix,
#     beta_par,
#     device = "cuda"
# ) {
#   
#   X <- as.matrix(X)
#   A_matrix <- as.matrix(A_matrix)
#   
#   N <- nrow(X)
#   d <- ncol(X)
#   
#   # ----------------------------------------------------------
#   # Unpack A exactly as in CPU implementation
#   # ----------------------------------------------------------
#   
#   unpacked <- logvMEM_unpack_A(
#     A_matrix,
#     d = d
#   )
#   
#   p_local <- unpacked$p
#   A_matrix_list <- unpacked$A_list
#   
#   if (N <= p_local) {
#     stop("Need N > p.")
#   }
#   
#   # ----------------------------------------------------------
#   # Keep these calculations identical to CPU version for now
#   # ----------------------------------------------------------
#   
#   mu_vec <- colMeans(X)
#   variance_vec <- diag(cov(X))
#   
#   M <- rep(
#     digamma(beta_par),
#     d
#   ) - log(beta_par)
#   
#   omega <- crossprod(
#     t(
#       diag(1, d) -
#         Reduce(`+`, A_matrix_list)
#     ),
#     matrix(
#       log(mu_vec) -
#         variance_vec / (2 * mu_vec^2)
#     )
#   ) - M
#   
#   omega <- as.numeric(omega)
#   
#   # ----------------------------------------------------------
#   # Build lag design exactly as CPU implementation
#   # ----------------------------------------------------------
#   
#   log_x <- log(X)
#   
#   lag_blocks <- lapply(
#     seq_len(p_local),
#     function(lag_id) {
#       
#       log_x[
#         (p_local + 1L - lag_id):(N - lag_id),
#         ,
#         drop = FALSE
#       ]
#     }
#   )
#   
#   lag_design <- do.call(
#     cbind,
#     lag_blocks
#   )
#   
#   # ==========================================================
#   # CUDA SECTION
#   # ==========================================================
#   
#   dtype <- torch::torch_float32()
#   
#   lag_t <- torch::torch_tensor(
#     lag_design,
#     dtype = dtype,
#     device = device
#   )
#   
#   A_t <- torch::torch_tensor(
#     A_matrix,
#     dtype = dtype,
#     device = device
#   )
#   
#   omega_t <- torch::torch_tensor(
#     omega,
#     dtype = dtype,
#     device = device
#   )
#   
#   # Same operation as:
#   #
#   # lag_design %*% t(A_matrix)
#   #
#   log_mu_t <- torch::torch_matmul(
#     lag_t,
#     A_t$t()
#   )
#   
#   # Add omega to every row
#   log_mu_t <- log_mu_t + omega_t
#   
#   mu_t <- torch::torch_exp(log_mu_t)
#   
#   list(
#     log_mu_t = log_mu_t,
#     mu_t = mu_t,
#     omega = omega,
#     p = p_local,
#     device = device
#   )
# }

# ============================================================
# END-TO-END CUDA LOG-vMEM LIKELIHOOD
#
# log(mu), mu, epsilon and MV-Gamma likelihood are evaluated
# on CUDA. Only the final scalar likelihood is returned to CPU.
# ============================================================

# log_lik_logvMEM_cuda_full <- function(
    #     X,
#     A_matrix,
#     beta_par,
#     n_quad = 64L,
#     device = "cuda",
#     context = NULL
# ) {
#   
#   X <- as.matrix(X)
#   A_matrix <- as.matrix(A_matrix)
#   
#   N <- nrow(X)
#   d <- ncol(X)
#   
#   # ----------------------------------------------------------
#   # Parameter checks
#   # ----------------------------------------------------------
#   
#   if (
#     !is.finite(beta_par) ||
#     beta_par <= 0
#   ) {
#     return(-Inf)
#   }
#   
#   # ==========================================================
#   # 1. COMPUTE log(mu) AND mu ON CUDA
#   # ==========================================================
#   
#   cm <- compute_log_mu_logvMEM_cuda(
#     X = X,
#     A_matrix = A_matrix,
#     beta_par = beta_par,
#     device = device
#   )
#   
#   p_local <- cm$p
#   
#   # ==========================================================
#   # 2. MOVE X_use TO CUDA
#   # ==========================================================
#   
#   X_use <- X[
#     (p_local + 1L):N,
#     ,
#     drop = FALSE
#   ]
#   
#   X_use_t <- torch::torch_tensor(
#     X_use,
#     dtype = torch::torch_float32(),
#     device = device
#   )
#   
#   # ==========================================================
#   # 3. COMPUTE MULTIPLICATIVE RESIDUALS ON CUDA
#   #
#   # epsilon_t = X_t / mu_t
#   # ==========================================================
#   
#   eps_t <- X_use_t / cm$mu_t
#   
#   # ==========================================================
#   # 4. CREATE MV-GAMMA CONTEXT IF NOT SUPPLIED
#   #
#   # alpha  = beta/2
#   # lambda = beta/2
#   # theta  = beta
#   # ==========================================================
#   
#   alpha_par <- beta_par / 2
#   
#   lambda_vec <- rep(
#     beta_par / 2,
#     d
#   )
#   
#   theta_vec <- rep(
#     beta_par,
#     d
#   )
#   
#   if (is.null(context)) {
#     
#     context <- create_mvgamma_gpu_context(
#       alpha = alpha_par,
#       lambda_vec = lambda_vec,
#       n_quad = n_quad,
#       device = device
#     )
#   }
#   
#   # ==========================================================
#   # 5. MV-GAMMA LOG-LIKELIHOOD ON CUDA
#   # ==========================================================
#   
#   logdens_sum <- mvgamma_loglik_gpu_tensor(
#     x_t = eps_t,
#     beta = beta_par,
#     theta_vec = theta_vec,
#     context = context
#   )
#   
#   if (!is.finite(logdens_sum)) {
#     return(-Inf)
#   }
#   
#   # ==========================================================
#   # 6. JACOBIAN TERM
#   #
#   # Existing likelihood:
#   #
#   # sum(logdens) - sum(log_mu)
#   #
#   # Sum log_mu ON CUDA and return only scalar.
#   # ==========================================================
#   
#   log_mu_sum <- as.numeric(
#     torch::torch_sum(
#       cm$log_mu_t
#     )$to(device = "cpu")
#   )
#   
#   # ==========================================================
#   # FINAL LOG-LIKELIHOOD
#   # ==========================================================
#   
#   logdens_sum - log_mu_sum
# }

# ============================================================
# CREATE PERSISTENT CUDA CONTEXT FOR log-vMEM
#
# Static quantities associated with X, p and beta are
# constructed once and kept resident on the GPU.
# ============================================================

create_logvMEM_cuda_context <- function(
    X,
    p,
    beta_par,
    n_quad = 64L,
    device = "cuda"
) {
  
  X <- as.matrix(X)
  
  N <- nrow(X)
  d <- ncol(X)
  
  p <- as.integer(p)
  
  if (N <= p) {
    stop("Need N > p.")
  }
  
  if (!is.finite(beta_par) || beta_par <= 0) {
    stop("beta_par must be positive and finite.")
  }
  
  dtype <- torch::torch_float32()
  
  # ==========================================================
  # 1. STATIC CPU QUANTITIES
  # ==========================================================
  
  mu_vec <- colMeans(X)
  variance_vec <- diag(cov(X))
  
  base_vec <- (
    log(mu_vec) -
      variance_vec / (2 * mu_vec^2)
  )
  
  M <- rep(
    digamma(beta_par),
    d
  ) - log(beta_par)
  
  # ==========================================================
  # 2. BUILD LAG DESIGN ONCE
  # ==========================================================
  
  log_x <- log(X)
  
  lag_blocks <- lapply(
    seq_len(p),
    function(lag_id) {
      
      log_x[
        (p + 1L - lag_id):(N - lag_id),
        ,
        drop = FALSE
      ]
    }
  )
  
  lag_design <- do.call(
    cbind,
    lag_blocks
  )
  
  X_use <- X[
    (p + 1L):N,
    ,
    drop = FALSE
  ]
  
  # ==========================================================
  # 3. MOVE STATIC OBJECTS TO CUDA ONCE
  # ==========================================================
  
  lag_t <- torch::torch_tensor(
    lag_design,
    dtype = dtype,
    device = device
  )
  
  X_use_t <- torch::torch_tensor(
    X_use,
    dtype = dtype,
    device = device
  )
  
  base_t <- torch::torch_tensor(
    base_vec,
    dtype = dtype,
    device = device
  )
  
  M_t <- torch::torch_tensor(
    M,
    dtype = dtype,
    device = device
  )
  
  # ==========================================================
  # 4. MV-GAMMA CONTEXT
  #
  # beta is fixed during an A-block optimization.
  # ==========================================================
  
  alpha_par <- beta_par / 2
  
  lambda_vec <- rep(
    beta_par / 2,
    d
  )
  
  theta_vec <- rep(
    beta_par,
    d
  )
  
  mvgamma_context <- create_mvgamma_gpu_context(
    alpha = alpha_par,
    lambda_vec = lambda_vec,
    n_quad = n_quad,
    device = device
  )
  
  # ==========================================================
  # RETURN PERSISTENT CONTEXT
  # ==========================================================
  
  structure(
    list(
      N = N,
      d = d,
      p = p,
      
      beta_par = beta_par,
      n_quad = n_quad,
      
      dtype = dtype,
      device = device,
      
      # GPU-resident static tensors
      lag_t = lag_t,
      X_use_t = X_use_t,
      base_t = base_t,
      M_t = M_t,
      
      # CPU values needed for bookkeeping
      base_vec = base_vec,
      M = M,
      theta_vec = theta_vec,
      
      # Persistent MV-Gamma context
      mvgamma_context = mvgamma_context
    ),
    class = "logvMEM_cuda_context"
  )
}

# ============================================================
# RESIDENT CUDA log-vMEM LIKELIHOOD FOR A OPTIMIZATION
#
# X, lag design, X_use, M, base vector, quadrature rules,
# and beta-dependent MV-Gamma quantities are already resident.
#
# Only candidate A is transferred to CUDA.
# ============================================================

log_lik_logvMEM_cuda_context <- function(
    A_matrix,
    context
) {
  
  if (!inherits(context, "logvMEM_cuda_context")) {
    stop(
      "context must be created by create_logvMEM_cuda_context()."
    )
  }
  
  A_matrix <- as.matrix(A_matrix)
  
  d <- context$d
  p <- context$p
  
  if (
    nrow(A_matrix) != d ||
    ncol(A_matrix) != d * p
  ) {
    stop("A_matrix has incompatible dimensions.")
  }
  
  if (any(!is.finite(A_matrix))) {
    return(-Inf)
  }
  
  # ==========================================================
  # 1. CANDIDATE A -> CUDA
  # ==========================================================
  
  A_t <- torch::torch_tensor(
    A_matrix,
    dtype = context$dtype,
    device = context$device
  )
  
  # ==========================================================
  # 2. COMPUTE sum_l A_l
  #
  # A = [A1 A2 ... Ap]
  # ==========================================================
  
  A_sum_t <- torch::torch_zeros(
    c(d, d),
    dtype = context$dtype,
    device = context$device
  )
  
  for (lag_id in seq_len(p)) {
    
    cols <- (
      ((lag_id - 1L) * d + 1L):
        (lag_id * d)
    )
    
    A_sum_t <- A_sum_t + A_t[, cols]
  }
  
  # ==========================================================
  # 3. COMPUTE omega ON CUDA
  #
  # CPU reference:
  #
  # crossprod(
  #   t(I - sum A_l),
  #   matrix(base_vec)
  # ) - M
  #
  # Algebraically:
  #
  # (I - sum A_l) %*% base_vec - M
  # ==========================================================
  
  I_t <- torch::torch_eye(
    d,
    dtype = context$dtype,
    device = context$device
  )
  
  omega_t <- torch::torch_matmul(
    I_t - A_sum_t,
    context$base_t$unsqueeze(2)
  )$squeeze(2) - context$M_t
  
  # ==========================================================
  # 4. COMPUTE log(mu)
  #
  # lag_design %*% t(A) + omega
  # ==========================================================
  
  log_mu_t <- torch::torch_matmul(
    context$lag_t,
    A_t$t()
  )
  
  log_mu_t <- log_mu_t + omega_t
  
  # ==========================================================
  # 5. mu AND MULTIPLICATIVE RESIDUALS ON CUDA
  # ==========================================================
  
  mu_t <- torch::torch_exp(log_mu_t)
  
  eps_t <- context$X_use_t / mu_t
  
  # ==========================================================
  # 6. MV-GAMMA LIKELIHOOD
  # ==========================================================
  
  logdens_sum <- mvgamma_loglik_gpu_tensor(
    x_t = eps_t,
    beta = context$beta_par,
    theta_vec = context$theta_vec,
    context = context$mvgamma_context
  )
  
  if (!is.finite(logdens_sum)) {
    return(-Inf)
  }
  
  # ==========================================================
  # 7. JACOBIAN TERM
  # ==========================================================
  
  log_mu_sum <- as.numeric(
    torch::torch_sum(
      log_mu_t
    )$to(device = "cpu")
  )
  
  # ==========================================================
  # FINAL SCALAR
  # ==========================================================
  
  logdens_sum - log_mu_sum
}
############################################################
# CPU REFERENCE likelihood.
# Keeps the trusted adaptive mvgamma_pdf()/integrate() calculation.
############################################################
log_lik_logvMEM_cpu_reference <- function(X, A_matrix, beta_par) {
  X <- as.matrix(X)
  A_matrix <- as.matrix(A_matrix)
  N <- nrow(X)
  d <- ncol(X)
  
  cm <- compute_log_mu_logvMEM_vectorized(
    X = X,
    A_matrix = A_matrix,
    beta_par = beta_par
  )
  p_local <- cm$p
  log_mu_use <- cm$log_mu
  mu_use <- cm$mu
  X_use <- X[(p_local + 1L):N, , drop = FALSE]
  
  alpha <- beta_par / 2
  beta <- beta_par
  lambda_vec <- rep(beta_par / 2, d)
  theta_vec <- rep(beta_par, d)
  
  ll <- 0
  for (row_id in seq_len(nrow(X_use))) {
    eps_t <- X_use[row_id, ] / mu_use[row_id, ]
    logdens <- mvgamma_pdf(
      x_vec = eps_t,
      alpha = alpha,
      beta = beta,
      lambda_vec = lambda_vec,
      theta_vec = theta_vec,
      log = TRUE
    )
    if (!is.finite(logdens)) return(-Inf)
    ll <- ll + logdens - sum(log_mu_use[row_id, ])
  }
  ll
}

############################################################
# CPU fixed-quadrature likelihood (double precision).
# Useful for isolating quadrature error from GPU/FP32 error.
############################################################
log_lik_logvMEM_cpu_quadrature <- function(
    X,
    A_matrix,
    beta_par,
    n_quad = 64L
) {
  X <- as.matrix(X)
  A_matrix <- as.matrix(A_matrix)
  N <- nrow(X)
  d <- ncol(X)
  
  cm <- compute_log_mu_logvMEM_vectorized(X, A_matrix, beta_par)
  p_local <- cm$p
  X_use <- X[(p_local + 1L):N, , drop = FALSE]
  eps_mat <- X_use / cm$mu
  
  logdens <- mvgamma_logpdf_batch_cpu_quadrature(
    x_mat = eps_mat,
    alpha = beta_par / 2,
    beta = beta_par,
    lambda_vec = rep(beta_par / 2, d),
    theta_vec = rep(beta_par, d),
    n_quad = n_quad
  )
  
  if (any(!is.finite(logdens))) return(-Inf)
  sum(logdens) - sum(cm$log_mu)
}

############################################################
# GPU likelihood (CUDA / Apple MPS).
# Fixed quadrature is evaluated using batched float32
# torch operations on the selected GPU backend.
############################################################
# log_lik_logvMEM_gpu <- function(
    #     X,
#     A_matrix,
#     beta_par,
#     n_quad = 64L,
#     device = NULL
# ) {
#   X <- as.matrix(X)
#   A_matrix <- as.matrix(A_matrix)
#   N <- nrow(X)
#   d <- ncol(X)
# 
#   cm <- compute_log_mu_logvMEM_vectorized(X, A_matrix, beta_par)
#   p_local <- cm$p
#   X_use <- X[(p_local + 1L):N, , drop = FALSE]
#   eps_mat <- X_use / cm$mu
# 
#   logdens <- mvgamma_logpdf_batch_gpu(
#     x_mat = eps_mat,
#     alpha = beta_par / 2,
#     beta = beta_par,
#     lambda_vec = rep(beta_par / 2, d),
#     theta_vec = rep(beta_par, d),
#     n_quad = n_quad,
#     device = device
#   )
# 
#   if (any(!is.finite(logdens))) return(-Inf)
#   sum(logdens) - sum(cm$log_mu)
# }

############################################################
# Public dispatcher: SAME name/signature as the existing project.
# Default is CPU reference, so merely sourcing this test file does not
# silently change the estimator. Set LOGVMEM_LIKELIHOOD_BACKEND only when
# you deliberately want another backend.
############################################################
# log_lik_logvMEM <- function(X, A_matrix, beta_par) {
#   backend <- tolower(Sys.getenv(
#     "LOGVMEM_LIKELIHOOD_BACKEND",
#     unset = "cpu_reference"
#   ))
#   n_quad <- suppressWarnings(as.integer(Sys.getenv(
#     "LOGVMEM_N_QUAD",
#     unset = "64"
#   )))
#   if (!is.finite(n_quad) || n_quad < 2L) n_quad <- 64L
# 
#   if (backend %in% c("cpu", "cpu_reference", "reference")) {
#     return(log_lik_logvMEM_cpu_reference(X, A_matrix, beta_par))
#   }
#   if (backend %in% c("cpu_quadrature", "quadrature")) {
#     return(log_lik_logvMEM_cpu_quadrature(X, A_matrix, beta_par, n_quad = n_quad))
#   }
#   if (backend %in% c("gpu", "cuda", "mps")) {
#     
#     device <- switch(
#       backend,
#       "cuda" = "cuda",
#       "mps"  = "mps",
#       "gpu"  = NULL
#     )
#     
#     return(
#       log_lik_logvMEM_gpu(
#         X,
#         A_matrix,
#         beta_par,
#         n_quad = n_quad,
#         device = device
#       )
#     )
#   }
# 
#   stop(
#     "Unknown LOGVMEM_LIKELIHOOD_BACKEND='", backend,
#     "'. Use cpu_reference, cpu_quadrature, gpu, cuda, or mps."
#   )
# }

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
# update_A_block <- function(X, A_current, i.index, beta_par,
#                            lower = -0.5, upper = 1,
#                            max_iter_bobyqa = 1000, eps = 1e-4) {
#   d <- ncol(X)
#   p <- ncol(A_current) / d
#   
#   stopifnot(nrow(A_current) == d)
#   stopifnot(ncol(A_current) == p * d)
#   
#   lb <- rep(lower, p * d)
#   ub <- rep(upper, p * d)
#   
#   obj_fun <- function(a_i_vec) {
#     A_trial <- A_current
#     A_trial[i.index, ] <- a_i_vec
#     val <- -log_lik_logvMEM(X = X, A_matrix = A_trial, beta_par = beta_par)
#     
#     if (!is.finite(val)) val <- 1e12
#     val
#   }
#   #BOBYQA performs derivative-free bound-constrained optimization using an iteratively constructed quadratic approximation for the objective function.
#   fit <- tryCatch(
#     bobyqa(
#       x0 = as.numeric(A_current[i.index, ]),
#       fn = obj_fun,
#       lower = lb,
#       upper = ub,
#       control = list(maxeval = max_iter_bobyqa, xtol_rel = eps)
#     ),
#     error = function(e) NULL
#   )
#   
#   if (is.null(fit) || is.null(fit$par) || any(!is.finite(fit$par))) {
#     return(list(
#       A_i_estimate = as.numeric(A_current[i.index, ]),
#       convergence = NA,
#       iter = NA,
#       objective = obj_fun(as.numeric(A_current[i.index, ]))
#     ))
#   }
#   
#   list(
#     A_i_estimate = fit$par,
#     convergence = fit$convergence,
#     iter = fit$iter,
#     objective = fit$value
#   )
# }
############################################################
# Update one row/block of A while holding others fixed
#
# Uses a persistent CUDA context created outside this
# function. BOBYQA remains on the CPU, while each likelihood
# evaluation is performed using the resident CUDA likelihood.
############################################################

update_A_block <- function(
    A_current,
    i.index,
    cuda_context,
    lower = -0.5,
    upper = 1,
    max_iter_bobyqa = 1000,
    eps = 1e-4
) {
  
  d <- cuda_context$d
  p <- cuda_context$p
  
  A_current <- as.matrix(A_current)
  
  stopifnot(nrow(A_current) == d)
  stopifnot(ncol(A_current) == p * d)
  
  lb <- rep(lower, p * d)
  ub <- rep(upper, p * d)
  
  # ==========================================================
  # BOBYQA objective
  #
  # Only row i.index changes.
  # X, beta, lag design, quadrature rules, etc. are already
  # contained in cuda_context.
  # ==========================================================
  
  obj_fun <- function(a_i_vec) {
    
    A_trial <- A_current
    A_trial[i.index, ] <- a_i_vec
    
    loglik <- log_lik_logvMEM_cuda_context(
      A_matrix = A_trial,
      context = cuda_context
    )
    
    val <- -loglik
    
    if (!is.finite(val)) {
      val <- 1e12
    }
    
    val
  }
  
  # ==========================================================
  # BOBYQA remains CPU-side
  # ==========================================================
  
  fit <- tryCatch(
    bobyqa(
      x0 = as.numeric(A_current[i.index, ]),
      fn = obj_fun,
      lower = lb,
      upper = ub,
      control = list(
        maxeval = max_iter_bobyqa,
        xtol_rel = eps
      )
    ),
    error = function(e) NULL
  )
  
  # ==========================================================
  # Fallback if optimization fails
  # ==========================================================
  
  if (
    is.null(fit) ||
    is.null(fit$par) ||
    any(!is.finite(fit$par))
  ) {
    
    return(
      list(
        A_i_estimate = as.numeric(A_current[i.index, ]),
        convergence = NA,
        iter = NA,
        objective = obj_fun(
          as.numeric(A_current[i.index, ])
        )
      )
    )
  }
  
  # ==========================================================
  # Return updated row
  # ==========================================================
  
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
# update_A_blockwise <- function(X, A_init, beta_par,
#                                max_block_sweeps = 1,
#                                lower = -0.5, upper = 1,
#                                max_iter_bobyqa = 1000, eps = 1e-4,
#                                trace = TRUE) {
#   A_current <- A_init
#   d <- ncol(X)
#   
#   block_log <- vector("list", max_block_sweeps)
#   
#   for (sweep in seq_len(max_block_sweeps)) {
#     if (trace) cat("  A-block sweep:", sweep, "\n")
#     
#     row_log <- vector("list", d)
#     
#     for (i in seq_len(d)) {
#       upd <- update_A_block(
#         X = X,
#         A_current = A_current,
#         i.index = i,
#         beta_par = beta_par,
#         lower = lower,
#         upper = upper,
#         max_iter_bobyqa = max_iter_bobyqa,
#         eps = eps
#       )
#       
#       A_current[i, ] <- upd$A_i_estimate
#       row_log[[i]] <- upd
#     }
#     
#     ll_now <- log_lik_logvMEM(X = X, A_matrix = A_current, beta_par = beta_par)
#     
#     block_log[[sweep]] <- list(
#       sweep = sweep,
#       loglik = ll_now,
#       row_updates = row_log
#     )
#     
#     if (trace) cat("    loglik after sweep =", ll_now, "\n")
#   }
#   
#   list(
#     A_estimate = A_current,
#     loglik = log_lik_logvMEM(X = X, A_matrix = A_current, beta_par = beta_par),
#     block_log = block_log
#   )
# }
############################################################
# One or more full Gauss-Seidel sweeps over rows of A
#
# A single persistent CUDA context is created for the entire
# A-update because X, p and beta remain fixed throughout all
# rows and all block sweeps.
############################################################

update_A_blockwise <- function(
    X,
    A_init,
    beta_par,
    max_block_sweeps = 1,
    lower = -0.5,
    upper = 1,
    max_iter_bobyqa = 1000,
    eps = 1e-4,
    n_quad = 64L,
    device = "cuda",
    trace = TRUE
) {
  
  X <- as.matrix(X)
  A_current <- as.matrix(A_init)
  
  d <- ncol(X)
  p <- ncol(A_current) / d
  
  stopifnot(nrow(A_current) == d)
  stopifnot(ncol(A_current) == p * d)
  
  # ==========================================================
  # CREATE CUDA CONTEXT ONCE
  #
  # beta is fixed during the entire A-update.
  # Therefore this context is reused for:
  #
  #   every BOBYQA evaluation
  #   every row
  #   every block sweep
  #
  # ==========================================================
  
  cuda_context <- create_logvMEM_cuda_context(
    X = X,
    p = p,
    beta_par = beta_par,
    n_quad = n_quad,
    device = device
  )
  
  if (trace) {
    cat(
      "  CUDA context created:",
      "d =", d,
      "| p =", p,
      "| beta =", beta_par,
      "| Q =", n_quad,
      "\n"
    )
  }
  
  block_log <- vector(
    "list",
    max_block_sweeps
  )
  
  # ==========================================================
  # BLOCK SWEEPS
  # ==========================================================
  
  for (sweep in seq_len(max_block_sweeps)) {
    
    if (trace) {
      cat(
        "  A-block sweep:",
        sweep,
        "\n"
      )
    }
    
    row_log <- vector(
      "list",
      d
    )
    
    # ========================================================
    # GAUSS-SEIDEL ROW UPDATES
    # ========================================================
    
    for (i in seq_len(d)) {
      
      upd <- update_A_block(
        A_current = A_current,
        i.index = i,
        cuda_context = cuda_context,
        lower = lower,
        upper = upper,
        max_iter_bobyqa = max_iter_bobyqa,
        eps = eps
      )
      
      # Gauss-Seidel update:
      # immediately use the newly estimated row
      A_current[i, ] <- upd$A_i_estimate
      
      row_log[[i]] <- upd
    }
    
    # ========================================================
    # LIKELIHOOD AFTER COMPLETE SWEEP
    # ========================================================
    
    ll_now <- log_lik_logvMEM_cuda_context(
      A_matrix = A_current,
      context = cuda_context
    )
    
    block_log[[sweep]] <- list(
      sweep = sweep,
      loglik = ll_now,
      row_updates = row_log
    )
    
    if (trace) {
      cat(
        "    loglik after sweep =",
        ll_now,
        "\n"
      )
    }
  }
  
  # ==========================================================
  # FINAL LIKELIHOOD
  # ==========================================================
  
  final_loglik <- log_lik_logvMEM_cuda_context(
    A_matrix = A_current,
    context = cuda_context
  )
  
  # ==========================================================
  # RETURN
  # ==========================================================
  
  list(
    A_estimate = A_current,
    loglik = final_loglik,
    block_log = block_log
  )
}
############################################################
# Update beta while holding A fixed
# 1D optimization
############################################################
# update_beta_1d <- function(X, A_matrix,
#                            beta_lower = 0.2,
#                            beta_upper = 20,
#                            tol = 1e-3) {
#   obj_fun <- function(beta_par) {
#     if (!is.finite(beta_par) || beta_par <= 0) return(1e12)
#     
#     val <- -log_lik_logvMEM(X = X, A_matrix = A_matrix, beta_par = beta_par)
#     if (!is.finite(val)) val <- 1e12
#     val
#   }
#   #The function optimize searches the interval from lower to upper for a minimum or maximum of the function f with respect to its first argument.
#   fit <- tryCatch(
#     optimize(f = obj_fun, interval = c(beta_lower, beta_upper), tol = tol),
#     error = function(e) NULL
#   )
#   
#   if (is.null(fit)) {
#     beta_fallback <- (beta_lower + beta_upper) / 2
#     return(list(
#       beta_estimate = beta_fallback,
#       objective = obj_fun(beta_fallback)
#     ))
#   }
#   
#   list(
#     beta_estimate = fit$minimum,
#     objective = fit$objective
#   )
# }
############################################################
# Update beta while holding A fixed
#
# Each candidate beta requires a new CUDA context because:
#
#   M        = digamma(beta) - log(beta)
#   alpha    = beta / 2
#   lambda_j = beta / 2
#   theta_j  = beta
#
# therefore the MV-Gamma quadrature context changes with beta.
############################################################

update_beta_1d <- function(
    X,
    A_matrix,
    beta_lower = 0.2,
    beta_upper = 20,
    tol = 1e-3,
    n_quad = 64L,
    device = "cuda"
) {
  
  X <- as.matrix(X)
  A_matrix <- as.matrix(A_matrix)
  
  d <- ncol(X)
  p <- ncol(A_matrix) / d
  
  stopifnot(nrow(A_matrix) == d)
  stopifnot(ncol(A_matrix) == p * d)
  
  # ==========================================================
  # OBJECTIVE FOR optimize()
  #
  # A is fixed.
  # beta changes at each evaluation.
  # ==========================================================
  
  obj_fun <- function(beta_par) {
    
    if (
      !is.finite(beta_par) ||
      beta_par <= 0
    ) {
      return(1e12)
    }
    
    # --------------------------------------------------------
    # New context required for this candidate beta
    # --------------------------------------------------------
    
    cuda_context <- tryCatch(
      create_logvMEM_cuda_context(
        X = X,
        p = p,
        beta_par = beta_par,
        n_quad = n_quad,
        device = device
      ),
      error = function(e) NULL
    )
    
    if (is.null(cuda_context)) {
      return(1e12)
    }
    
    # --------------------------------------------------------
    # Evaluate likelihood for fixed A and candidate beta
    # --------------------------------------------------------
    
    loglik <- tryCatch(
      log_lik_logvMEM_cuda_context(
        A_matrix = A_matrix,
        context = cuda_context
      ),
      error = function(e) -Inf
    )
    
    val <- -loglik
    
    if (!is.finite(val)) {
      val <- 1e12
    }
    
    val
  }
  
  # ==========================================================
  # 1D BOUNDED OPTIMIZATION
  # ==========================================================
  
  fit <- tryCatch(
    optimize(
      f = obj_fun,
      interval = c(
        beta_lower,
        beta_upper
      ),
      tol = tol
    ),
    error = function(e) NULL
  )
  
  # ==========================================================
  # FALLBACK
  # ==========================================================
  
  if (is.null(fit)) {
    
    beta_fallback <- (
      beta_lower + beta_upper
    ) / 2
    
    return(
      list(
        beta_estimate = beta_fallback,
        objective = obj_fun(beta_fallback)
      )
    )
  }
  
  # ==========================================================
  # RETURN
  # ==========================================================
  
  list(
    beta_estimate = fit$minimum,
    objective = fit$objective
  )
}
############################################################
# Main Method 1 estimator
# Alternating optimization of A and beta
############################################################
# estimate_method1_joint <- function(X,
#                                    init_A_matrix,
#                                    init_beta,
#                                    max_outer_iter = 10,
#                                    max_block_sweeps = 1,
#                                    lower_A = -0.5,
#                                    upper_A = 1,
#                                    beta_lower = 0.2,
#                                    beta_upper = 20,
#                                    max_iter_bobyqa = 1000,
#                                    eps_A = 1e-4,
#                                    tol_beta = 1e-3,
#                                    tol_outer_A = 1e-6,
#                                    tol_outer_beta = 1e-6,
#                                    trace = TRUE) {
#   A_current <- init_A_matrix
#   beta_current <- init_beta
#   
#   if (!is.matrix(A_current)) A_current <- as.matrix(A_current)
#   if (!is.numeric(beta_current) || length(beta_current) != 1 || beta_current <= 0) {
#     stop("init_beta must be a positive scalar.")
#   }
#   
#   history <- vector("list", max_outer_iter)
#   
#   ll_current <- log_lik_logvMEM(
#     X = X,
#     A_matrix = A_current,
#     beta_par = beta_current
#   )
#   
#   if (trace) {
#     cat("Initial beta =", beta_current, "\n")
#     cat("Initial loglik =", ll_current, "\n")
#   }
#   
#   for (iter in seq_len(max_outer_iter)) {
#     if (trace) cat("\nOuter iteration:", iter, "\n")
#     
#     A_old <- A_current
#     beta_old <- beta_current
#     
#     ##################################################
#     # Step 1: update A blockwise given beta
#     ##################################################
#     
#     ll_before_A <- ll_current
#     
#     A_upd <- update_A_blockwise(
#       X = X,
#       A_init = A_current,
#       beta_par = beta_current,
#       max_block_sweeps = max_block_sweeps,
#       lower = lower_A,
#       upper = upper_A,
#       max_iter_bobyqa = max_iter_bobyqa,
#       eps = eps_A,
#       trace = trace
#     )
#     
#     A_current <- A_upd$A_estimate
#     
#     ll_after_A <- log_lik_logvMEM(
#       X = X,
#       A_matrix = A_current,
#       beta_par = beta_current
#     )
#     
#     delta_ll_A_rel <- abs(ll_after_A - ll_before_A) /
#       max(1, abs(ll_before_A))
#     
#     ##################################################
#     # Step 2: update beta given updated A
#     ##################################################
#     
#     beta_upd <- update_beta_1d(
#       X = X,
#       A_matrix = A_current,
#       beta_lower = beta_lower,
#       beta_upper = beta_upper,
#       tol = tol_beta
#     )
#     
#     beta_current <- beta_upd$beta_estimate
#     
#     ll_new <- log_lik_logvMEM(
#       X = X,
#       A_matrix = A_current,
#       beta_par = beta_current
#     )
#     
#     delta_ll_beta_rel <- abs(ll_new - ll_after_A) /
#       max(1, abs(ll_after_A))
#     
#     ##################################################
#     # Diagnostics: keep old parameter changes too
#     ##################################################
#     
#     delta_A <- sqrt(sum((A_current - A_old)^2))
#     delta_beta <- abs(beta_current - beta_old)
#     
#     history[[iter]] <- list(
#       iter = iter,
#       A = A_current,
#       beta = beta_current,
#       loglik = ll_new,
#       delta_A = delta_A,
#       delta_beta = delta_beta,
#       delta_ll_A_rel = delta_ll_A_rel,
#       delta_ll_beta_rel = delta_ll_beta_rel,
#       A_update_log = A_upd$block_log,
#       beta_objective = beta_upd$objective
#     )
#     
#     if (trace) {
#       cat("  Updated beta =", beta_current, "\n")
#       cat("  Updated loglik =", ll_new, "\n")
#       cat("  delta_A =", delta_A, "\n")
#       cat("  delta_beta =", delta_beta, "\n")
#       cat("  delta_ll_A_rel =", delta_ll_A_rel, "\n")
#       cat("  delta_ll_beta_rel =", delta_ll_beta_rel, "\n")
#     }
#     
#     ##################################################
#     # Stopping rule: relative likelihood change
#     ##################################################
#     
#     if (delta_ll_A_rel < tol_outer_A &&
#         delta_ll_beta_rel < tol_outer_beta) {
#       
#       if (trace) {
#         cat("Converged based on relative log-likelihood change.\n")
#       }
#       
#       history <- history[seq_len(iter)]
#       
#       return(list(
#         A_estimate = A_current,
#         beta_estimate = beta_current,
#         loglik = ll_new,
#         n_outer_iter = iter,
#         converged = TRUE,
#         history = history
#       ))
#     }
#     
#     ll_current <- ll_new
#   }
#   
#   list(
#     A_estimate = A_current,
#     beta_estimate = beta_current,
#     loglik = ll_current,
#     n_outer_iter = max_outer_iter,
#     converged = FALSE,
#     history = history
#   )
# }

############################################################
# Joint estimation of A and beta
#
# Alternating optimization:
#
#   1. Update A | beta fixed
#   2. Update beta | A fixed
#
# Uses resident CUDA likelihood for A optimization and
# beta-specific CUDA contexts for beta optimization.
############################################################

estimate_method1_joint <- function(
    X,
    p,
    A_init = NULL,
    beta_init = 5,
    #max_outer_iter = 20,
    max_block_sweeps = 1,
    max_iter_bobyqa = 1000,
    lower_A = -0.5,
    upper_A = 1,
    beta_lower = 0.2,
    beta_upper = 20,
    eps_A = 1e-4,
    tol_beta = 1e-3,
    max_outer_iter = 10,
    tol_rel_loglik = 1e-5,
    rel_loglik_patience = 2L,
    n_quad = 32L,
    tol_outer_A = 1e-3,
    tol_outer_beta = 1e-3,
    #n_quad = 64L,
    device = "cuda",
    trace = TRUE
) {
  
  X <- as.matrix(X)
  
  d <- ncol(X)
  
  # ==========================================================
  # INITIAL VALUES
  # ==========================================================
  
  if (is.null(A_init)) {
    
    A_current <- matrix(
      0.01,
      nrow = d,
      ncol = p * d
    )
    
  } else {
    
    A_current <- as.matrix(A_init)
    
  }
  
  stopifnot(nrow(A_current) == d)
  stopifnot(ncol(A_current) == p * d)
  
  beta_current <- beta_init
  
  
  # ==========================================================
  # INITIAL LOG-LIKELIHOOD
  # ==========================================================
  
  initial_context <- create_logvMEM_cuda_context(
    X = X,
    p = p,
    beta_par = beta_current,
    n_quad = n_quad,
    device = device
  )
  
  ll_current <- log_lik_logvMEM_cuda_context(
    A_matrix = A_current,
    context = initial_context
  )
  
  if (trace) {
    
    cat("\n========================================\n")
    cat("JOINT LOG-vMEM ESTIMATION\n")
    cat("========================================\n")
    
    cat(
      "Initial beta =",
      beta_current,
      "\n"
    )
    
    cat(
      "Initial loglik =",
      ll_current,
      "\n"
    )
  }
  
  
  # ==========================================================
  # STORAGE
  # ==========================================================
  
  iteration_log <- vector(
    "list",
    max_outer_iter
  )
  
  converged <- FALSE
  
  
  # ==========================================================
  # OUTER ALTERNATING LOOP
  # ==========================================================
  stable_loglik_count <- 0L
  for (outer in seq_len(max_outer_iter)) {
    
    if (trace) {
      
      cat(
        "\n----------------------------------------\n"
      )
      
      cat(
        "Outer iteration:",
        outer,
        "\n"
      )
      
      cat(
        "Starting beta:",
        beta_current,
        "\n"
      )
      
      cat(
        "Starting loglik:",
        ll_current,
        "\n"
      )
    }
    
    
    # Save parameters before this outer iteration
    A_old <- A_current
    beta_old <- beta_current
    
    
    # ========================================================
    # STEP 1: UPDATE A | beta fixed
    # ========================================================
    
    A_upd <- update_A_blockwise(
      X = X,
      A_init = A_current,
      beta_par = beta_current,
      max_block_sweeps = max_block_sweeps,
      lower = lower_A,
      upper = upper_A,
      max_iter_bobyqa = max_iter_bobyqa,
      eps = eps_A,
      n_quad = n_quad,
      device = device,
      trace = trace
    )
    
    A_current <- A_upd$A_estimate
    
    # Already calculated using the same fixed-beta CUDA context
    ll_after_A <- A_upd$loglik
    
    
    if (trace) {
      
      cat(
        "  loglik after A update =",
        ll_after_A,
        "\n"
      )
    }
    
    
    # ========================================================
    # STEP 2: UPDATE beta | A fixed
    # ========================================================
    
    beta_upd <- update_beta_1d(
      X = X,
      A_matrix = A_current,
      beta_lower = beta_lower,
      beta_upper = beta_upper,
      tol = tol_beta,
      n_quad = n_quad,
      device = device
    )
    
    beta_current <- beta_upd$beta_estimate
    
    
    # ========================================================
    # LOG-LIKELIHOOD AFTER BOTH UPDATES
    #
    # beta changed, so create a fresh beta-specific context.
    # ========================================================
    
    new_context <- create_logvMEM_cuda_context(
      X = X,
      p = p,
      beta_par = beta_current,
      n_quad = n_quad,
      device = device
    )
    
    ll_new <- log_lik_logvMEM_cuda_context(
      A_matrix = A_current,
      context = new_context
    )
    
    #Relative loglikelihood
    ll_old <- ll_current
    
    rel_loglik_change <- abs(
      ll_new - ll_old
    ) / max(1, abs(ll_old))
    
    if (trace) {
      cat(
        "  relative loglik change =",
        rel_loglik_change,
        "\n"
      )
    }
    
    if (trace) {
      
      cat(
        "  updated beta =",
        beta_current,
        "\n"
      )
      
      cat(
        "  loglik after beta update =",
        ll_new,
        "\n"
      )
    }
    
    
    # ========================================================
    # CONVERGENCE MEASURES
    # ========================================================
    
    A_change <- max(
      abs(A_current - A_old)
    )
    
    beta_change <- abs(
      beta_current - beta_old
    )
    
    
    if (trace) {
      
      cat(
        "  max |delta A| =",
        A_change,
        "\n"
      )
      
      cat(
        "  |delta beta| =",
        beta_change,
        "\n"
      )
    }
    
    
    # ========================================================
    # STORE ITERATION
    # ========================================================
    
    iteration_log[[outer]] <- list(
      iteration = outer,
      A_estimate = A_current,
      beta_estimate = beta_current,
      loglik_before = ll_old,
      loglik_after_A = ll_after_A,
      loglik = ll_new,
      rel_loglik_change = rel_loglik_change,
      A_change = A_change,
      beta_change = beta_change,
      A_block_log = A_upd$block_log
    )
    
    
    # ========================================================
    # UPDATE CURRENT LIKELIHOOD
    # ========================================================
    
    ll_current <- ll_new
    
    
    # ========================================================
    # CHECK CONVERGENCE
    # ========================================================
    
    if (rel_loglik_change < tol_rel_loglik) {
      stable_loglik_count <- stable_loglik_count + 1L
    } else {
      stable_loglik_count <- 0L
    }
    
    if (stable_loglik_count >= rel_loglik_patience) {
      
      converged <- TRUE
      
      if (trace) {
        cat(
          "\nConvergence reached at outer iteration",
          outer,
          "after",
          stable_loglik_count,
          "consecutive stable log-likelihood iterations\n"
        )
      }
      
      break
    }
  }
  
  
  # ==========================================================
  # TRIM ITERATION LOG
  # ==========================================================
  
  n_iterations <- outer
  
  iteration_log <- iteration_log[
    seq_len(n_iterations)
  ]
  
  
  # ==========================================================
  # FINAL OUTPUT
  # ==========================================================
  
  if (trace) {
    
    cat("\n========================================\n")
    cat("ESTIMATION COMPLETE\n")
    cat("========================================\n")
    
    cat(
      "Iterations:",
      n_iterations,
      "\n"
    )
    
    cat(
      "Converged:",
      converged,
      "\n"
    )
    
    cat(
      "Final beta:",
      beta_current,
      "\n"
    )
    
    cat(
      "Final loglik:",
      ll_current,
      "\n"
    )
    
    cat(
      "========================================\n"
    )
  }
  
  
  return(
    list(
      A_estimate = A_current,
      beta_estimate = beta_current,
      loglik = ll_current,
      convergence = converged,
      iterations = n_iterations,
      iteration_log = iteration_log
    )
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
