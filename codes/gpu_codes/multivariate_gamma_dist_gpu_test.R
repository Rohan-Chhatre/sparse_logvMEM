
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







############################################################
# GPU / fixed-quadrature helpers for corrected MVGamma
#
# UPDATED: Gauss-Jacobi quadrature
#
# The original mvgamma_pdf() above is UNCHANGED and remains
# the trusted adaptive-integrate() reference.
#
# After z = B*u, with B = min(x), the integrand contains
#
#   u^(alpha - 1) (1-u)^(lambda_j - 1)
#
# where j is the component attaining B.
# Gauss-Jacobi quadrature absorbs these endpoint powers into
# the quadrature weights, which is especially important when
# alpha < 1 and/or lambda_j < 1.
#
# Function names are intentionally unchanged so that
# mle_logvMEM_1_gpu_test.R does NOT need to change.
############################################################

.mvgamma_gj_cache <- new.env(parent = emptyenv())

############################################################
# Gauss-Jacobi rule on [0,1]
#
# Approximates
#
#   integral_0^1 u^b (1-u)^a f(u) du
#
# by
#
#   sum_q w_q f(u_q)
#
# for a > -1 and b > -1.
############################################################

gauss_jacobi_rule_01 <- function(n = 64L, a, b) {
  
  n <- as.integer(n)
  
  if (!is.finite(n) || n < 2L) {
    stop("n must be an integer >= 2")
  }
  
  if (!is.finite(a) || !is.finite(b) || a <= -1 || b <= -1) {
    stop("Gauss-Jacobi parameters a and b must both be > -1.")
  }
  
  key <- paste0(
    "n", n,
    "_a", formatC(a, digits = 14, format = "fg", flag = "#"),
    "_b", formatC(b, digits = 14, format = "fg", flag = "#")
  )
  
  if (exists(key, envir = .mvgamma_gj_cache, inherits = FALSE)) {
    return(get(key, envir = .mvgamma_gj_cache, inherits = FALSE))
  }
  
  # Golub-Welsch construction for the standard Jacobi weight
  # (1-x)^a (1+x)^b on [-1,1].
  k0 <- 0:(n - 1L)
  
  denom_diag <- (
    (2 * k0 + a + b) *
      (2 * k0 + a + b + 2)
  )
  
  diag_j <- (b^2 - a^2) / denom_diag
  
  # Stable limiting form for the k=0 coefficient when a+b = 0.
  if (abs(a + b) < 1e-12) {
    diag_j[1] <- (b - a) / (a + b + 2)
  }
  
  k <- 1:(n - 1L)
  
  ############################################################
  # Robust off-diagonal Jacobi recurrence
  #
  # The usual formula contains the ratio
  #
  #   (k + a + b) / (2*k + a + b - 1)
  #
  # At beta = 1:
  #
  #   alpha = lambda = 0.5
  #   a = b = -0.5
  #
  # and for k = 1 both numerator and denominator are zero.
  # This is a removable singularity; its limiting value is 1.
  ############################################################
  
  s <- a + b
  
  ratio_num <- k + s
  ratio_den <- 2 * k + s - 1
  
  ratio <- ratio_num / ratio_den
  
  removable <- (
    abs(ratio_num) < 1e-12 &
      abs(ratio_den) < 1e-12
  )
  
  ratio[removable] <- 1
  
  off_sq <- (
    4 * k *
      (k + a) *
      (k + b) *
      ratio /
      (
        (2 * k + s)^2 *
          (2 * k + s + 1)
      )
  )
  
  # Protect against tiny negative values caused purely by
  # floating-point roundoff.
  off_sq[
    off_sq < 0 &
      off_sq > -1e-14
  ] <- 0
  
  if (
    any(!is.finite(off_sq)) ||
    any(off_sq < 0)
  ) {
    stop(
      "Invalid Gauss-Jacobi recurrence coefficient ",
      "for a = ", a,
      ", b = ", b
    )
  }
  
  off_j <- sqrt(off_sq)
  
  J <- diag(diag_j, nrow = n, ncol = n)
  
  J[cbind(1:(n - 1L), 2:n)] <- off_j
  J[cbind(2:n, 1:(n - 1L))] <- off_j
  
  ee <- eigen(J, symmetric = TRUE)
  ord <- order(ee$values)
  
  # Transform [-1,1] nodes to [0,1].
  nodes <- (ee$values[ord] + 1) / 2
  
  # The weights on [0,1] sum to Beta(b+1, a+1).
  weights <- beta(b + 1, a + 1) * (ee$vectors[1, ord]^2)
  
  out <- list(
    nodes = nodes,
    weights = weights
  )
  
  assign(key, out, envir = .mvgamma_gj_cache)
  
  out
}


############################################################
# Index of the component attaining B = min(x)
############################################################

mvgamma_min_index <- function(x_mat) {
  max.col(-x_mat, ties.method = "first")
}


############################################################
# Batched Gauss-Jacobi quadrature on CPU (FP64)
#
# The helper name is kept unchanged so the MLE test file and
# stress-test script can be reused without modification.
############################################################

mvgamma_logpdf_batch_cpu_quadrature <- function(
    x_mat,
    alpha,
    beta,
    lambda_vec,
    theta_vec,
    n_quad = 64L
) {
  
  x_mat <- as.matrix(x_mat)
  storage.mode(x_mat) <- "double"
  
  lambda_vec <- as.numeric(lambda_vec)
  theta_vec <- as.numeric(theta_vec)
  
  n <- nrow(x_mat)
  d <- ncol(x_mat)
  
  if (length(lambda_vec) != d) {
    stop("length(lambda_vec) must equal ncol(x_mat)")
  }
  
  if (length(theta_vec) != d) {
    stop("length(theta_vec) must equal ncol(x_mat)")
  }
  
  if (
    !is.finite(alpha) ||
    !is.finite(beta) ||
    alpha <= 0 ||
    beta <= 0 ||
    any(!is.finite(lambda_vec)) ||
    any(!is.finite(theta_vec)) ||
    any(lambda_vec <= 0) ||
    any(theta_vec <= 0)
  ) {
    return(rep(-Inf, n))
  }
  
  bad_row <- apply(
    x_mat,
    1L,
    function(x) any(!is.finite(x)) || any(x <= 0)
  )
  
  out <- rep(-Inf, n)
  good <- which(!bad_row)
  
  if (length(good) == 0L) {
    return(out)
  }
  
  X <- x_mat[good, , drop = FALSE]
  B <- apply(X, 1L, min)
  min_index <- mvgamma_min_index(X)
  
  log_const <- (
    alpha * log(beta) -
      lgamma(alpha) +
      sum(lambda_vec * log(theta_vec) - lgamma(lambda_vec))
  )
  
  log_pdf_good <- rep(NA_real_, length(good))
  
  # At most d groups, depending on which coordinate attains B.
  for (j in seq_len(d)) {
    
    ids <- which(min_index == j)
    
    if (length(ids) == 0L) {
      next
    }
    
    Xg <- X[ids, , drop = FALSE]
    Bg <- B[ids]
    
    lambda_min <- lambda_vec[j]
    
    # u^(alpha-1) (1-u)^(lambda_j-1)
    rule <- gauss_jacobi_rule_01(
      n = n_quad,
      a = lambda_min - 1,
      b = alpha - 1
    )
    
    u <- rule$nodes
    log_w <- log(rule$weights)
    
    ng <- length(ids)
    Q <- length(u)
    
    # z = B*u
    z <- Bg %o% u
    
    # Smooth part after the endpoint power factors have been
    # absorbed into the Jacobi weights.
    log_smooth <- -beta * z
    
    for (k in seq_len(d)) {
      
      shifted <- (
        matrix(Xg[, k], nrow = ng, ncol = Q) -
          z
      )
      
      shifted[shifted <= 0] <- .Machine$double.xmin
      
      if (k == j) {
        
        # (B-z)^(lambda_j-1) is already represented by the
        # (1-u)^(lambda_j-1) Jacobi weight.
        log_smooth <- (
          log_smooth -
            theta_vec[k] * shifted
        )
        
      } else {
        
        log_smooth <- (
          log_smooth +
            (lambda_vec[k] - 1) * log(shifted) -
            theta_vec[k] * shifted
        )
      }
    }
    
    # z^(alpha-1), (B-z)^(lambda_j-1), and dz contribute
    #
    # B^(alpha + lambda_j - 1).
    log_B_factor <- (
      alpha + lambda_min - 1
    ) * log(Bg)
    
    log_terms <- (
      log_smooth +
        matrix(
          log_w,
          nrow = ng,
          ncol = Q,
          byrow = TRUE
        )
    )
    
    # Stable row-wise log-sum-exp.
    row_max <- apply(log_terms, 1L, max)
    
    log_integral <- (
      log_B_factor +
        row_max +
        log(
          rowSums(
            exp(log_terms - row_max)
          )
        )
    )
    
    log_pdf_good[ids] <- log_const + log_integral
  }
  
  out[good] <- log_pdf_good
  out
}


############################################################
# Detect whether Apple MPS can execute torch tensors
############################################################

torch_mps_available_logvmem <- function() {
  
  if (!requireNamespace("torch", quietly = TRUE)) {
    return(FALSE)
  }
  
  ok <- tryCatch(
    {
      x <- torch::torch_tensor(
        1,
        dtype = torch::torch_float32(),
        device = "mps"
      )
      
      y <- x + 1
      
      invisible(
        as.numeric(
          y$to(device = "cpu")
        )
      )
      
      TRUE
    },
    error = function(e) FALSE
  )
  
  isTRUE(ok)
}

torch_cuda_available_logvmem <- function() {
  
  if (!requireNamespace("torch", quietly = TRUE)) {
    return(FALSE)
  }
  
  isTRUE(torch::cuda_is_available())
}
############################################################
# Torch device selection
#
# Priority: NVIDIA CUDA -> Apple MPS -> CPU
############################################################
get_torch_device_logvmem <- function() {
  
  if (torch_cuda_available_logvmem()) {
    return("cuda")
  }
  
  if (torch_mps_available_logvmem()) {
    return("mps")
  }
  
  return("cpu")
}
# mvgamma_logpdf_batch_gpu <- function(
    #     x_mat,
#     alpha,
#     beta,
#     lambda_vec,
#     theta_vec,
#     n_quad = 64L,
#     device = NULL
# ) {
#   if (!requireNamespace("torch", quietly = TRUE)) {
#     stop(
#       "Package 'torch' is required for GPU likelihood evaluation."
#     )
#   }
#   
#   if (is.null(device)) {
#     device <- get_torch_device_logvmem()
#   }
#   x_mat <- as.matrix(x_mat)
#   storage.mode(x_mat) <- "double"
# 
#   lambda_vec <- as.numeric(lambda_vec)
#   theta_vec <- as.numeric(theta_vec)
# 
#   n <- nrow(x_mat)
#   d <- ncol(x_mat)
# 
#   if (length(lambda_vec) != d) {
#     stop("length(lambda_vec) must equal ncol(x_mat)")
#   }
# 
#   if (length(theta_vec) != d) {
#     stop("length(theta_vec) must equal ncol(x_mat)")
#   }
# 
#   if (
#     !is.finite(alpha) ||
#     !is.finite(beta) ||
#     alpha <= 0 ||
#     beta <= 0 ||
#     any(!is.finite(lambda_vec)) ||
#     any(!is.finite(theta_vec)) ||
#     any(lambda_vec <= 0) ||
#     any(theta_vec <= 0)
#   ) {
#     return(rep(-Inf, n))
#   }
# 
#   if (
#     identical(device, "cuda") &&
#     !torch_cuda_available_logvmem()
#   ) {
#     stop("CUDA GPU is not available.")
#   }
#   
#   if (
#     identical(device, "mps") &&
#     !torch_mps_available_logvmem()
#   ) {
#     stop("Apple MPS GPU is not available.")
#   }
# 
#   bad_row <- apply(
#     x_mat,
#     1L,
#     function(x) any(!is.finite(x)) || any(x <= 0)
#   )
# 
#   out <- rep(-Inf, n)
#   good <- which(!bad_row)
# 
#   if (length(good) == 0L) {
#     return(out)
#   }
# 
#   Xr <- x_mat[good, , drop = FALSE]
#   Br <- apply(Xr, 1L, min)
#   min_index <- mvgamma_min_index(Xr)
# 
#   dtype <- torch::torch_float32()
# 
#   # Keep special-function constant in R FP64.
#   log_const <- (
#     alpha * log(beta) -
#     lgamma(alpha) +
#     sum(lambda_vec * log(theta_vec) - lgamma(lambda_vec))
#   )
# 
#   log_pdf_good <- rep(NA_real_, length(good))
# 
#   for (j in seq_len(d)) {
# 
#     ids <- which(min_index == j)
# 
#     if (length(ids) == 0L) {
#       next
#     }
# 
#     Xg_r <- Xr[ids, , drop = FALSE]
#     Bg_r <- Br[ids]
# 
#     lambda_min <- lambda_vec[j]
# 
#     rule <- gauss_jacobi_rule_01(
#       n = n_quad,
#       a = lambda_min - 1,
#       b = alpha - 1
#     )
# 
#     Xg <- torch::torch_tensor(
#       Xg_r,
#       dtype = dtype,
#       device = device
#     )
# 
#     Bg <- torch::torch_tensor(
#       Bg_r,
#       dtype = dtype,
#       device = device
#     )$unsqueeze(2)
# 
#     u <- torch::torch_tensor(
#       rule$nodes,
#       dtype = dtype,
#       device = device
#     )$unsqueeze(1)
# 
#     log_w <- torch::torch_tensor(
#       log(rule$weights),
#       dtype = dtype,
#       device = device
#     )$unsqueeze(1)
# 
#     # n_group x Q
#     z <- Bg * u
# 
#     # n_group x Q x d
#     X3 <- Xg$unsqueeze(2)
#     z3 <- z$unsqueeze(3)
# 
#     shifted <- X3 - z3
# 
#     shifted <- torch::torch_clamp(
#       shifted,
#       min = 1e-30
#     )
# 
#     lambda_t <- torch::torch_tensor(
#       lambda_vec,
#       dtype = dtype,
#       device = device
#     )$view(c(1, 1, d))
# 
#     theta_t <- torch::torch_tensor(
#       theta_vec,
#       dtype = dtype,
#       device = device
#     )$view(c(1, 1, d))
# 
#     log_component <- (
#       (lambda_t - 1) * torch::torch_log(shifted) -
#       theta_t * shifted
#     )
# 
#     # Remove the power term for the minimum component because
#     # it has already been absorbed into the Jacobi weight.
#     log_component[, , j] <- (
#       -theta_t[, , j] * shifted[, , j]
#     )
# 
#     log_smooth <- (
#       torch::torch_sum(
#         log_component,
#         dim = 3
#       ) -
#       beta * z
#     )
# 
#     log_terms <- log_smooth + log_w
# 
#     log_integral_core <- torch::torch_logsumexp(
#       log_terms,
#       dim = 2
#     )
# 
#     log_B_factor <- (
#       alpha + lambda_min - 1
#     ) * torch::torch_log(
#       Bg$squeeze(2)
#     )
# 
#     log_pdf_gpu <- (
#       log_integral_core +
#       log_B_factor +
#       log_const
#     )
# 
#     log_pdf_good[ids] <- as.numeric(
#       log_pdf_gpu$to(device = "cpu")
#     )
#   }
# 
#   out[good] <- log_pdf_good
#   out
# }


# ============================================================
# RESIDENT / REDUCED-TRANSFER GPU MV-GAMMA LOG-PDF
# ============================================================

# mvgamma_logpdf_batch_gpu_resident <- function(
    #     x_mat,
#     alpha,
#     beta,
#     lambda_vec,
#     theta_vec,
#     n_quad = 64L,
#     device = NULL
# ) {
#   if (!requireNamespace("torch", quietly = TRUE)) {
#     stop(
#       "Package 'torch' is required for GPU likelihood evaluation."
#     )
#   }
#   
#   if (is.null(device)) {
#     device <- get_torch_device_logvmem()
#   }
#   
#   x_mat <- as.matrix(x_mat)
#   storage.mode(x_mat) <- "double"
#   
#   lambda_vec <- as.numeric(lambda_vec)
#   theta_vec  <- as.numeric(theta_vec)
#   
#   n <- nrow(x_mat)
#   d <- ncol(x_mat)
#   
#   if (length(lambda_vec) != d) {
#     stop("length(lambda_vec) must equal ncol(x_mat)")
#   }
#   
#   if (length(theta_vec) != d) {
#     stop("length(theta_vec) must equal ncol(x_mat)")
#   }
#   
#   if (
#     !is.finite(alpha) ||
#     !is.finite(beta) ||
#     alpha <= 0 ||
#     beta <= 0 ||
#     any(!is.finite(lambda_vec)) ||
#     any(!is.finite(theta_vec)) ||
#     any(lambda_vec <= 0) ||
#     any(theta_vec <= 0)
#   ) {
#     return(rep(-Inf, n))
#   }
#   
#   if (
#     identical(device, "cuda") &&
#     !torch_cuda_available_logvmem()
#   ) {
#     stop("CUDA GPU is not available.")
#   }
#   
#   if (
#     identical(device, "mps") &&
#     !torch_mps_available_logvmem()
#   ) {
#     stop("Apple MPS GPU is not available.")
#   }
#   
#   # ----------------------------------------------------------
#   # Identify valid rows on CPU
#   # ----------------------------------------------------------
#   
#   bad_row <- apply(
#     x_mat,
#     1L,
#     function(x) any(!is.finite(x)) || any(x <= 0)
#   )
#   
#   out <- rep(-Inf, n)
#   good <- which(!bad_row)
#   
#   if (length(good) == 0L) {
#     return(out)
#   }
#   
#   Xr <- x_mat[good, , drop = FALSE]
#   Br <- apply(Xr, 1L, min)
#   min_index <- mvgamma_min_index(Xr)
#   
#   dtype <- torch::torch_float32()
#   
#   # ----------------------------------------------------------
#   # Special-function constant calculated in R FP64
#   # ----------------------------------------------------------
#   
#   log_const <- (
#     alpha * log(beta) -
#       lgamma(alpha) +
#       sum(
#         lambda_vec * log(theta_vec) -
#           lgamma(lambda_vec)
#       )
#   )
#   
#   # ==========================================================
#   # TRANSFER REUSABLE OBJECTS TO GPU ONCE
#   # ==========================================================
#   
#   Xr_t <- torch::torch_tensor(
#     Xr,
#     dtype = dtype,
#     device = device
#   )
#   
#   Br_t <- torch::torch_tensor(
#     Br,
#     dtype = dtype,
#     device = device
#   )
#   
#   lambda_t <- torch::torch_tensor(
#     lambda_vec,
#     dtype = dtype,
#     device = device
#   )$view(c(1, 1, d))
#   
#   theta_t <- torch::torch_tensor(
#     theta_vec,
#     dtype = dtype,
#     device = device
#   )$view(c(1, 1, d))
#   
#   # Keep output on GPU until all groups have been evaluated.
#   log_pdf_good_t <- torch::torch_empty(
#     length(good),
#     dtype = dtype,
#     device = device
#   )
#   
#   # ----------------------------------------------------------
#   # Cache GPU quadrature tensors within this likelihood call.
#   #
#   # Components having the same lambda_min use the same
#   # Gauss-Jacobi rule, so nodes and weights need only be
#   # transferred once for each unique rule.
#   # ----------------------------------------------------------
#   
#   quadrature_cache <- new.env(
#     hash = TRUE,
#     parent = emptyenv()
#   )
#   
#   # ==========================================================
#   # PROCESS MINIMUM-COMPONENT GROUPS
#   # ==========================================================
#   
#   for (j in seq_len(d)) {
#     
#     ids <- which(min_index == j)
#     
#     if (length(ids) == 0L) {
#       next
#     }
#     
#     # --------------------------------------------------------
#     # Construct GPU indices.
#     #
#     # index_select preserves dimensions even when a group
#     # contains only one observation.
#     # --------------------------------------------------------
#     
#     ids_t <- torch::torch_tensor(
#       ids,
#       dtype = torch::torch_int64(),
#       device = device
#     )
#     
#     Xg <- Xr_t$index_select(
#       dim = 1,
#       index = ids_t
#     )
#     
#     Bg <- Br_t$index_select(
#       dim = 1,
#       index = ids_t
#     )$unsqueeze(2)
#     
#     lambda_min <- lambda_vec[j]
#     
#     # --------------------------------------------------------
#     # Obtain/cache Gauss-Jacobi rule on GPU
#     # --------------------------------------------------------
#     
#     cache_key <- paste(
#       format(lambda_min, digits = 17),
#       format(alpha, digits = 17),
#       n_quad,
#       sep = "_"
#     )
#     
#     if (!exists(
#       cache_key,
#       envir = quadrature_cache,
#       inherits = FALSE
#     )) {
#       
#       rule <- gauss_jacobi_rule_01(
#         n = n_quad,
#         a = lambda_min - 1,
#         b = alpha - 1
#       )
#       
#       u_t <- torch::torch_tensor(
#         rule$nodes,
#         dtype = dtype,
#         device = device
#       )$unsqueeze(1)
#       
#       log_w_t <- torch::torch_tensor(
#         log(rule$weights),
#         dtype = dtype,
#         device = device
#       )$unsqueeze(1)
#       
#       assign(
#         cache_key,
#         list(
#           u = u_t,
#           log_w = log_w_t
#         ),
#         envir = quadrature_cache
#       )
#     }
#     
#     quad <- get(
#       cache_key,
#       envir = quadrature_cache,
#       inherits = FALSE
#     )
#     
#     u     <- quad$u
#     log_w <- quad$log_w
#     
#     # ========================================================
#     # GAUSS-JACOBI QUADRATURE ON GPU
#     # ========================================================
#     
#     # n_group x Q
#     z <- Bg * u
#     
#     # n_group x Q x d
#     X3 <- Xg$unsqueeze(2)
#     z3 <- z$unsqueeze(3)
#     
#     shifted <- X3 - z3
#     
#     shifted <- torch::torch_clamp(
#       shifted,
#       min = 1e-30
#     )
#     
#     log_component <- (
#       (lambda_t - 1) *
#         torch::torch_log(shifted) -
#         theta_t * shifted
#     )
#     
#     # Remove the power term for the minimum component because
#     # it is already incorporated into the Gauss-Jacobi weight.
#     log_component[, , j] <- (
#       -theta_t[, , j] *
#         shifted[, , j]
#     )
#     
#     log_smooth <- (
#       torch::torch_sum(
#         log_component,
#         dim = 3
#       ) -
#         beta * z
#     )
#     
#     log_terms <- log_smooth + log_w
#     
#     log_integral_core <- torch::torch_logsumexp(
#       log_terms,
#       dim = 2
#     )
#     
#     log_B_factor <- (
#       alpha + lambda_min - 1
#     ) * torch::torch_log(
#       Bg$squeeze(2)
#     )
#     
#     log_pdf_gpu <- (
#       log_integral_core +
#         log_B_factor +
#         log_const
#     )
#     
#     # --------------------------------------------------------
#     # Keep group result on GPU.
#     # No GPU -> CPU transfer here.
#     # --------------------------------------------------------
#     
#     log_pdf_good_t$index_copy_(
#       dim = 1,
#       index = ids_t,
#       source = log_pdf_gpu
#     )
#   }
#   
#   # ==========================================================
#   # SINGLE GPU -> CPU TRANSFER
#   # ==========================================================
#   
#   log_pdf_good <- as.numeric(
#     log_pdf_good_t$to(device = "cpu")
#   )
#   
#   out[good] <- log_pdf_good
#   
#   out
# }


# ============================================================
# CREATE PERSISTENT CUDA QUADRATURE CONTEXT
# ============================================================

create_mvgamma_gpu_context <- function(
    alpha,
    lambda_vec,
    n_quad = 64L,
    device = NULL
) {
  
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("Package 'torch' is required.")
  }
  
  if (is.null(device)) {
    device <- get_torch_device_logvmem()
  }
  
  lambda_vec <- as.numeric(lambda_vec)
  d <- length(lambda_vec)
  
  dtype <- torch::torch_float32()
  
  # ----------------------------------------------------------
  # lambda vector stays on GPU
  # ----------------------------------------------------------
  
  lambda_t <- torch::torch_tensor(
    lambda_vec,
    dtype = dtype,
    device = device
  )$view(c(1, 1, d))
  
  # ----------------------------------------------------------
  # Construct each unique Gauss-Jacobi rule ONCE
  # ----------------------------------------------------------
  
  quadrature_cache <- new.env(
    hash = TRUE,
    parent = emptyenv()
  )
  
  for (j in seq_len(d)) {
    
    lambda_min <- lambda_vec[j]
    
    cache_key <- paste(
      format(lambda_min, digits = 17),
      format(alpha, digits = 17),
      n_quad,
      sep = "_"
    )
    
    if (!exists(
      cache_key,
      envir = quadrature_cache,
      inherits = FALSE
    )) {
      
      rule <- gauss_jacobi_rule_01(
        n = n_quad,
        a = lambda_min - 1,
        b = alpha - 1
      )
      
      u_t <- torch::torch_tensor(
        rule$nodes,
        dtype = dtype,
        device = device
      )$unsqueeze(1)
      
      log_w_t <- torch::torch_tensor(
        log(rule$weights),
        dtype = dtype,
        device = device
      )$unsqueeze(1)
      
      assign(
        cache_key,
        list(
          u = u_t,
          log_w = log_w_t
        ),
        envir = quadrature_cache
      )
    }
  }
  
  # ----------------------------------------------------------
  # Return persistent context
  # ----------------------------------------------------------
  
  structure(
    list(
      alpha = alpha,
      lambda_vec = lambda_vec,
      lambda_t = lambda_t,
      n_quad = n_quad,
      d = d,
      dtype = dtype,
      device = device,
      quadrature_cache = quadrature_cache
    ),
    class = "mvgamma_gpu_context"
  )
}


# ============================================================
# MV-GAMMA TOTAL LOG-LIKELIHOOD USING PERSISTENT GPU CONTEXT
#
# Returns ONE scalar:
#     sum_t log f(x_t)
#
# The summation is performed on the GPU.
# ============================================================

# mvgamma_loglik_gpu_context <- function(
    #     x_mat,
#     beta,
#     theta_vec,
#     context
# ) {
#   
#   if (!inherits(context, "mvgamma_gpu_context")) {
#     stop("context must be created by create_mvgamma_gpu_context().")
#   }
#   
#   # ----------------------------------------------------------
#   # Extract persistent objects
#   # ----------------------------------------------------------
#   
#   alpha      <- context$alpha
#   lambda_vec <- context$lambda_vec
#   lambda_t   <- context$lambda_t
#   n_quad     <- context$n_quad
#   d          <- context$d
#   dtype      <- context$dtype
#   device     <- context$device
#   quad_cache <- context$quadrature_cache
#   
#   # ----------------------------------------------------------
#   # Input checks
#   # ----------------------------------------------------------
#   
#   x_mat <- as.matrix(x_mat)
#   storage.mode(x_mat) <- "double"
#   
#   theta_vec <- as.numeric(theta_vec)
#   
#   n <- nrow(x_mat)
#   
#   if (ncol(x_mat) != d) {
#     stop("ncol(x_mat) must equal context$d.")
#   }
#   
#   if (length(theta_vec) != d) {
#     stop("length(theta_vec) must equal context$d.")
#   }
#   
#   if (
#     !is.finite(beta) ||
#     beta <= 0 ||
#     any(!is.finite(theta_vec)) ||
#     any(theta_vec <= 0)
#   ) {
#     return(-Inf)
#   }
#   
#   # ----------------------------------------------------------
#   # Check observations
#   # ----------------------------------------------------------
#   
#   bad_row <- apply(
#     x_mat,
#     1L,
#     function(x) any(!is.finite(x)) || any(x <= 0)
#   )
#   
#   # For the likelihood, an invalid observation makes the
#   # total likelihood invalid.
#   if (any(bad_row)) {
#     return(-Inf)
#   }
#   
#   # ----------------------------------------------------------
#   # CPU preprocessing
#   # ----------------------------------------------------------
#   
#   Br <- apply(x_mat, 1L, min)
#   min_index <- mvgamma_min_index(x_mat)
#   
#   # ----------------------------------------------------------
#   # Transfer changing objects to GPU ONCE PER LIKELIHOOD CALL
#   # ----------------------------------------------------------
#   
#   X_t <- torch::torch_tensor(
#     x_mat,
#     dtype = dtype,
#     device = device
#   )
#   
#   Br_t <- torch::torch_tensor(
#     Br,
#     dtype = dtype,
#     device = device
#   )
#   
#   theta_t <- torch::torch_tensor(
#     theta_vec,
#     dtype = dtype,
#     device = device
#   )$view(c(1, 1, d))
#   
#   # ----------------------------------------------------------
#   # Constant calculated in R FP64
#   # ----------------------------------------------------------
#   
#   log_const <- (
#     alpha * log(beta) -
#       lgamma(alpha) +
#       sum(
#         lambda_vec * log(theta_vec) -
#           lgamma(lambda_vec)
#       )
#   )
#   
#   # ----------------------------------------------------------
#   # GPU accumulator
#   #
#   # Only ONE scalar is ultimately returned to the CPU.
#   # ----------------------------------------------------------
#   
#   total_loglik_t <- torch::torch_zeros(
#     c(1),
#     dtype = dtype,
#     device = device
#   )
#   
#   # ==========================================================
#   # PROCESS MINIMUM-COMPONENT GROUPS
#   # ==========================================================
#   
#   for (j in seq_len(d)) {
#     
#     ids <- which(min_index == j)
#     
#     if (length(ids) == 0L) {
#       next
#     }
#     
#     ids_t <- torch::torch_tensor(
#       ids,
#       dtype = torch::torch_int64(),
#       device = device
#     )
#     
#     Xg <- X_t$index_select(
#       dim = 1,
#       index = ids_t
#     )
#     
#     Bg <- Br_t$index_select(
#       dim = 1,
#       index = ids_t
#     )$unsqueeze(2)
#     
#     lambda_min <- lambda_vec[j]
#     
#     # --------------------------------------------------------
#     # Retrieve already-resident quadrature rule
#     # --------------------------------------------------------
#     
#     cache_key <- paste(
#       format(lambda_min, digits = 17),
#       format(alpha, digits = 17),
#       n_quad,
#       sep = "_"
#     )
#     
#     if (!exists(
#       cache_key,
#       envir = quad_cache,
#       inherits = FALSE
#     )) {
#       stop(
#         "Required quadrature rule is missing from GPU context."
#       )
#     }
#     
#     quad <- get(
#       cache_key,
#       envir = quad_cache,
#       inherits = FALSE
#     )
#     
#     u     <- quad$u
#     log_w <- quad$log_w
#     
#     # --------------------------------------------------------
#     # Gauss-Jacobi quadrature on GPU
#     # --------------------------------------------------------
#     
#     # n_group x Q
#     z <- Bg * u
#     
#     # n_group x Q x d
#     X3 <- Xg$unsqueeze(2)
#     z3 <- z$unsqueeze(3)
#     
#     shifted <- X3 - z3
#     
#     shifted <- torch::torch_clamp(
#       shifted,
#       min = 1e-30
#     )
#     
#     log_component <- (
#       (lambda_t - 1) *
#         torch::torch_log(shifted) -
#         theta_t * shifted
#     )
#     
#     # Minimum component power already incorporated
#     # into Gauss-Jacobi weight.
#     log_component[, , j] <- (
#       -theta_t[, , j] *
#         shifted[, , j]
#     )
#     
#     log_smooth <- (
#       torch::torch_sum(
#         log_component,
#         dim = 3
#       ) -
#         beta * z
#     )
#     
#     log_terms <- log_smooth + log_w
#     
#     log_integral_core <- torch::torch_logsumexp(
#       log_terms,
#       dim = 2
#     )
#     
#     log_B_factor <- (
#       alpha + lambda_min - 1
#     ) * torch::torch_log(
#       Bg$squeeze(2)
#     )
#     
#     log_pdf_gpu <- (
#       log_integral_core +
#         log_B_factor +
#         log_const
#     )
#     
#     # --------------------------------------------------------
#     # Sum this entire group ON THE GPU
#     # --------------------------------------------------------
#     
#     total_loglik_t <- (
#       total_loglik_t +
#         torch::torch_sum(log_pdf_gpu)
#     )
#   }
#   
#   # ==========================================================
#   # Transfer ONE scalar GPU -> CPU
#   # ==========================================================
#   
#   as.numeric(
#     total_loglik_t$to(device = "cpu")
#   )
# }


# ============================================================
# MV-GAMMA TOTAL LOG-LIKELIHOOD
# INPUT x_t IS ALREADY A CUDA TENSOR
#
# Important:
#   - no x_mat CPU -> GPU transfer
#   - summation stays on GPU
#   - only final scalar returns to CPU
# ============================================================

mvgamma_loglik_gpu_tensor <- function(
    x_t,
    beta,
    theta_vec,
    context
) {
  
  if (!inherits(context, "mvgamma_gpu_context")) {
    stop("context must be created by create_mvgamma_gpu_context().")
  }
  
  alpha      <- context$alpha
  lambda_vec <- context$lambda_vec
  lambda_t   <- context$lambda_t
  n_quad     <- context$n_quad
  d          <- context$d
  dtype      <- context$dtype
  device     <- context$device
  quad_cache <- context$quadrature_cache
  
  theta_vec <- as.numeric(theta_vec)
  
  if (
    !is.finite(beta) ||
    beta <= 0 ||
    length(theta_vec) != d ||
    any(!is.finite(theta_vec)) ||
    any(theta_vec <= 0)
  ) {
    return(-Inf)
  }
  
  # ----------------------------------------------------------
  # Minimum of each observation, calculated ON GPU
  # ----------------------------------------------------------
  
  min_result <- torch::torch_min(
    x_t,
    dim = 2
  )
  
  Br_t <- min_result[[1]]
  
  # R torch returns 1-based component indices
  min_index <- as.array(
    min_result[[2]]$to(device = "cpu")
  )
  
  # ----------------------------------------------------------
  # theta -> GPU
  # ----------------------------------------------------------
  
  theta_t <- torch::torch_tensor(
    theta_vec,
    dtype = dtype,
    device = device
  )$view(c(1, 1, d))
  
  # ----------------------------------------------------------
  # Normalizing constant
  # ----------------------------------------------------------
  
  log_const <- (
    alpha * log(beta) -
      lgamma(alpha) +
      sum(
        lambda_vec * log(theta_vec) -
          lgamma(lambda_vec)
      )
  )
  
  total_loglik_t <- torch::torch_zeros(
    c(1),
    dtype = dtype,
    device = device
  )
  
  # ==========================================================
  # MINIMUM-COMPONENT GROUPS
  # ==========================================================
  
  for (j in seq_len(d)) {
    
    ids <- which(min_index == j)
    
    if (length(ids) == 0L) {
      next
    }
    
    ids_t <- torch::torch_tensor(
      ids,
      dtype = torch::torch_int64(),
      device = device
    )
    
    Xg <- x_t$index_select(
      dim = 1,
      index = ids_t
    )
    
    Bg <- Br_t$index_select(
      dim = 1,
      index = ids_t
    )$unsqueeze(2)
    
    lambda_min <- lambda_vec[j]
    
    cache_key <- paste(
      format(lambda_min, digits = 17),
      format(alpha, digits = 17),
      n_quad,
      sep = "_"
    )
    
    quad <- get(
      cache_key,
      envir = quad_cache,
      inherits = FALSE
    )
    
    u     <- quad$u
    log_w <- quad$log_w
    
    # --------------------------------------------------------
    # Gauss-Jacobi quadrature
    # --------------------------------------------------------
    
    z <- Bg * u
    
    X3 <- Xg$unsqueeze(2)
    z3 <- z$unsqueeze(3)
    
    shifted <- X3 - z3
    
    shifted <- torch::torch_clamp(
      shifted,
      min = 1e-30
    )
    
    log_component <- (
      (lambda_t - 1) *
        torch::torch_log(shifted) -
        theta_t * shifted
    )
    
    log_component[, , j] <- (
      -theta_t[, , j] *
        shifted[, , j]
    )
    
    log_smooth <- (
      torch::torch_sum(
        log_component,
        dim = 3
      ) -
        beta * z
    )
    
    log_terms <- log_smooth + log_w
    
    log_integral_core <- torch::torch_logsumexp(
      log_terms,
      dim = 2
    )
    
    log_B_factor <- (
      alpha + lambda_min - 1
    ) * torch::torch_log(
      Bg$squeeze(2)
    )
    
    log_pdf_gpu <- (
      log_integral_core +
        log_B_factor +
        log_const
    )
    
    total_loglik_t <- (
      total_loglik_t +
        torch::torch_sum(log_pdf_gpu)
    )
  }
  
  # ----------------------------------------------------------
  # ONE scalar GPU -> CPU
  # ----------------------------------------------------------
  
  as.numeric(
    total_loglik_t$to(device = "cpu")
  )
}