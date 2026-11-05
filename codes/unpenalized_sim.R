############################################################
# FULL MONTE CARLO SIMULATION STUDY FOR METHOD 2
# Blockwise profile likelihood estimation
############################################################

rm(list = ls())

############################################################
# Setup
############################################################
project_dir <- "/Users/rohan/Documents/Rohan/Finance paper/Sparse-Logarithmic-Vector-Multiplicative-Error-Model-with-Multivariate-Gamma-Errors-develop_RCodes/"
setwd(project_dir)

source("codes/mle_logvMEM_1.R")
source("codes/multivariate_gamma_dist.R")

library(nloptr)
library(Rsolnp)
library(ACDm)
library(xtable)

############################################################
# TRUE MODEL PARAMETERS
############################################################
d <- 5
p <- 1
q <- 0
r <- max(p, q)

beta_true  <- 10
omega_true <- rep(0.1, d)

############################################################
# TRUE A MATRIX
############################################################
A_true <- matrix(c(
  -0.052, 0.377, 0.131, 0.000, 0.000,
  0.000, 0.106, 0.000, 0.122, 0.000,
  0.000, 0.113, 0.056, 0.022, 0.000,
  0.000, 0.000, 0.266, 0.041, 0.365,
  0.357, 0.000, 0.000, 0.854, 0.077
), nrow = d, byrow = TRUE)

A_matrix_list_true <- list(A_true)

############################################################
# SIMULATION SETTINGS
############################################################
B <- 100
N <- 1500
burn_in <- 500

############################################################
# STORAGE OBJECTS
############################################################
beta_hat_vec <- rep(NA_real_, B)
A_hat_list   <- vector("list", B)
runtime_vec  <- rep(NA_real_, B)
best_id_vec  <- rep(NA_integer_, B)
loglik_vec   <- rep(NA_real_, B)

############################################################
# INITIAL VALUES FOR ESTIMATION
############################################################
A_init_list <- lapply(1:p, function(lag_id) 0.05 * diag(d))
A_init <- do.call(cbind, A_init_list)

############################################################
# BETA GRID
############################################################
# Option 1: fixed grid
beta_grid_fixed <- seq(5, 15, length.out = 21)

# Option 2: data-adaptive grid from init.val()
# Use inside the loop if preferred

############################################################
# MONTE CARLO LOOP
############################################################
for (b in 1:B) {
  
  cat("Replication:", b, "of", B, "\n")
  set.seed(1000 + b)
  
  ##########################################################
  # Simulate Data
  ##########################################################
  X <- sim_logvMEM(
    N = N,
    d = d,
    omega = omega_true,
    A_matrix_list = A_matrix_list_true,
    B_matrix_list = NULL,
    beta_par = beta_true,
    burn_in = burn_in
  )
  
  ##########################################################
  # Optional: adaptive beta grid from univariate init
  ##########################################################
  init_vals <- init.val(X = X, maxlag = p, n_beta = 21)
  beta_grid <- init_vals$beta_list_max_min
  # beta_grid <- beta_grid_fixed   # uncomment this instead if you want a fixed grid
  
  ##########################################################
  # Estimate Parameters: Method 2
  ##########################################################
  t1 <- Sys.time()
  
  fit <- estimate_method2_profile(
    X = X,
    init_A_matrix = A_init,
    beta_grid = beta_grid,
    max_block_sweeps = 2,
    lower_A = -0.5,
    upper_A = 1,
    max_iter_bobyqa = 800,
    eps_A = 1e-4,
    trace = FALSE
  )
  
  t2 <- Sys.time()
  
  ##########################################################
  # Store Results
  ##########################################################
  beta_hat_vec[b] <- fit$beta_estimate
  A_hat_list[[b]] <- fit$A_estimate
  runtime_vec[b]  <- as.numeric(difftime(t2, t1, units = "secs"))
  best_id_vec[b]  <- fit$best_id
  loglik_vec[b]   <- fit$loglik
}

############################################################
# FUNCTION TO BUILD SUMMARY TABLE
############################################################
make_sim_table <- function(beta_hat_vec, A_hat_list, beta_true, A_true, digits = 3) {
  
  B <- length(beta_hat_vec)
  A_true_vec <- as.vector(A_true)
  A_hat_mat <- do.call(rbind, lapply(A_hat_list, function(Ahat) as.vector(Ahat)))
  
  est_mat <- cbind(beta_hat_vec, A_hat_mat)
  true_vec <- c(beta_true, A_true_vec)
  
  nr <- nrow(A_true)
  nc <- ncol(A_true)
  
  A_names <- c()
  for (j in 1:nc) {
    for (i in 1:nr) {
      A_names <- c(A_names, paste0("$A_{", i, j, "}$"))
    }
  }
  
  param_names <- c("$\\beta$", A_names)
  
  mean_est <- colMeans(est_mat, na.rm = TRUE)
  mb <- mean_est - true_vec
  rmse <- sqrt(colMeans((t(t(est_mat) - true_vec))^2, na.rm = TRUE))
  sd_est <- apply(est_mat, 2, sd, na.rm = TRUE)
  
  out <- data.frame(
    Parameter = param_names,
    `True value` = round(true_vec, digits),
    Mean = round(mean_est, digits),
    MB = round(mb, digits),
    RMSE = round(rmse, digits),
    SD = round(sd_est, digits),
    check.names = FALSE
  )
  
  out
}

############################################################
# GENERATE FINAL TABLE
############################################################
sim_table_method2 <- make_sim_table(
  beta_hat_vec = beta_hat_vec,
  A_hat_list = A_hat_list,
  beta_true = beta_true,
  A_true = A_true
)

print(sim_table_method2)

############################################################
# EXPORT LATEX TABLE
############################################################
print(
  xtable(sim_table_method2, digits = 3),
  include.rownames = FALSE,
  sanitize.text.function = identity
)

############################################################
# ADDITIONAL DIAGNOSTICS
############################################################
cat("\n====================================\n")
cat("Average Runtime (sec):", mean(runtime_vec, na.rm = TRUE), "\n")
cat("Median Runtime (sec):", median(runtime_vec, na.rm = TRUE), "\n")
cat("Average Selected Grid Index:", mean(best_id_vec, na.rm = TRUE), "\n")
cat("Average Profile LogLik:", mean(loglik_vec, na.rm = TRUE), "\n")
cat("====================================\n")