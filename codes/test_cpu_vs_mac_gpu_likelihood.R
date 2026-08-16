############################################################
# TEST: corrected log-vMEM likelihood — CPU vs Apple MPS GPU
#
# This script does NOT run the penalized simulation.
# It generates ONE dataset and evaluates the SAME (X, A, beta)
# using:
#   1) current adaptive CPU reference
#   2) fixed Gauss-Legendre quadrature on CPU (FP64)
#   3) same fixed quadrature on Apple MPS GPU (FP32)
############################################################

rm(list = ls())

project_dir <- "/Users/rohan/Documents/Rohan/Finance paper/Sparse-Logarithmic-Vector-Multiplicative-Error-Model-with-Multivariate-Gamma-Errors-develop_RCodes/"
setwd(project_dir)

# ---- Files required in codes/ ----
required_files <- c(
  "codes/multivariate_gamma_dist_gpu_test.R",
  "codes/mle_logvMEM_1_gpu_test.R",
  "codes/sim_logvMEM.R",
  "codes/generate_autoregressive_matrices.R"
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop("Missing files: ", paste(missing_files, collapse = ", "))
}

if (!requireNamespace("torch", quietly = TRUE)) {
  stop(
    "R package 'torch' is not installed. Run install.packages('torch'), restart R, then rerun this script."
  )
}

cat("R architecture:", R.version$arch, "\n")
cat("Machine:", Sys.info()[["machine"]], "\n")
cat("torch version:", as.character(utils::packageVersion("torch")), "\n")

# Source the test implementation. mle_logvMEM_1_gpu_test.R also sources the
# corresponding multivariate gamma test file.
source("codes/mle_logvMEM_1_gpu_test.R")

mps_ok <- torch_mps_available_logvmem()
cat("Apple MPS available:", mps_ok, "\n")
if (!mps_ok) {
  stop(
    "MPS was not detected. This GPU test requires an Apple-Silicon Mac with an MPS-enabled torch installation."
  )
}

############################################################
# 1. Generate ONE representative dataset
############################################################
set.seed(12345)

d <- 5
p <- 10
q <- 0
beta_true <- 4
omega_true <- rep(0.1, d)

# Use N = 1500 to match the simulation study.
# For a very quick first smoke test, temporarily change N to 300.
N <- 1500
burn_in <- 500

L <- matrix(
  c(
    3, 3, 3, 3, 3,
    4, 4, 4, 4, 4,
    4, 4, 4, 4, 4,
    5, 5, 5, 5, 5,
    5, 5, 5, 5, 5
  ),
  nrow = d,
  ncol = d,
  byrow = TRUE
)

A_obj <- autoregressive_matrix(
  d = d,
  p = p,
  L = L,
  max_eigen_abs = 0.96
)
A_matrix_list <- A_obj$A_matrix_list
A_matrix <- do.call(cbind, A_matrix_list)

X <- sim_logvMEM(
  N = N,
  d = d,
  omega = omega_true,
  A_matrix_list = A_matrix_list,
  B_matrix_list = NULL,
  beta_par = beta_true,
  burn_in = burn_in
)
X <- as.matrix(X)

cat("\nDataset dimensions:", paste(dim(X), collapse = " x "), "\n")
cat("A dimensions:", paste(dim(A_matrix), collapse = " x "), "\n")
cat("beta:", beta_true, "\n")

############################################################
# 2. Warm up MPS
############################################################
# GPU kernels have one-time startup/dispatch overhead. Warm up before timing.
cat("\nWarming up Apple MPS...\n")
invisible(log_lik_logvMEM_gpu(
  X = X[1:300, , drop = FALSE],
  A_matrix = A_matrix,
  beta_par = beta_true,
  n_quad = 64L,
  device = "mps"
))
cat("Warm-up complete.\n")

############################################################
# 3. Evaluate EXACT SAME likelihood inputs
############################################################
n_quad <- 64L

cat("\n1/3: Current adaptive CPU reference...\n")
t_cpu_ref <- system.time({
  ll_cpu_ref <- log_lik_logvMEM_cpu_reference(
    X = X,
    A_matrix = A_matrix,
    beta_par = beta_true
  )
})

cat("2/3: Fixed-quadrature CPU FP64...\n")
t_cpu_quad <- system.time({
  ll_cpu_quad <- log_lik_logvMEM_cpu_quadrature(
    X = X,
    A_matrix = A_matrix,
    beta_par = beta_true,
    n_quad = n_quad
  )
})

cat("3/3: Fixed-quadrature Apple MPS FP32...\n")
t_gpu <- system.time({
  ll_gpu <- log_lik_logvMEM_gpu(
    X = X,
    A_matrix = A_matrix,
    beta_par = beta_true,
    n_quad = n_quad,
    device = "mps"
  )
})

############################################################
# 4. Accuracy + timing report
############################################################
ref_abs_diff <- abs(ll_cpu_quad - ll_cpu_ref)
gpu_abs_diff <- abs(ll_gpu - ll_cpu_quad)
gpu_vs_ref_abs_diff <- abs(ll_gpu - ll_cpu_ref)

ref_rel_diff <- ref_abs_diff / max(1, abs(ll_cpu_ref))
gpu_rel_diff <- gpu_abs_diff / max(1, abs(ll_cpu_quad))
gpu_vs_ref_rel_diff <- gpu_vs_ref_abs_diff / max(1, abs(ll_cpu_ref))

elapsed_ref <- unname(t_cpu_ref[["elapsed"]])
elapsed_cpu_quad <- unname(t_cpu_quad[["elapsed"]])
elapsed_gpu <- unname(t_gpu[["elapsed"]])

results <- data.frame(
  method = c(
    "CPU adaptive reference",
    paste0("CPU Gauss-Legendre Q=", n_quad, " FP64"),
    paste0("Apple MPS Gauss-Legendre Q=", n_quad, " FP32")
  ),
  loglik = c(ll_cpu_ref, ll_cpu_quad, ll_gpu),
  elapsed_seconds = c(elapsed_ref, elapsed_cpu_quad, elapsed_gpu)
)

cat("\n============================================================\n")
cat("RESULTS\n")
cat("============================================================\n")
print(results, row.names = FALSE, digits = 12)

cat("\nAccuracy diagnostics\n")
cat("CPU quadrature - adaptive CPU |absolute difference| =",
    format(ref_abs_diff, digits = 12), "\n")
cat("CPU quadrature - adaptive CPU relative difference =",
    format(ref_rel_diff, scientific = TRUE, digits = 6), "\n")
cat("GPU FP32 - CPU quadrature FP64 |absolute difference| =",
    format(gpu_abs_diff, digits = 12), "\n")
cat("GPU FP32 - CPU quadrature FP64 relative difference =",
    format(gpu_rel_diff, scientific = TRUE, digits = 6), "\n")
cat("GPU FP32 - adaptive CPU |absolute difference| =",
    format(gpu_vs_ref_abs_diff, digits = 12), "\n")
cat("GPU FP32 - adaptive CPU relative difference =",
    format(gpu_vs_ref_rel_diff, scientific = TRUE, digits = 6), "\n")

cat("\nSpeed diagnostics\n")
cat("Adaptive CPU / fixed CPU speed ratio =",
    round(elapsed_ref / elapsed_cpu_quad, 2), "x\n")
cat("Adaptive CPU / GPU speed ratio =",
    round(elapsed_ref / elapsed_gpu, 2), "x\n")
cat("Fixed CPU / GPU speed ratio =",
    round(elapsed_cpu_quad / elapsed_gpu, 2), "x\n")

############################################################
# 5. Optional dispatcher check
############################################################
# This verifies that the SAME public function log_lik_logvMEM() can switch
# backend without changing any caller (e.g., BOBYQA or penalized code).
Sys.setenv(LOGVMEM_N_QUAD = as.character(n_quad))

Sys.setenv(LOGVMEM_LIKELIHOOD_BACKEND = "cpu_reference")
ll_dispatch_cpu <- log_lik_logvMEM(X, A_matrix, beta_true)

Sys.setenv(LOGVMEM_LIKELIHOOD_BACKEND = "gpu")
ll_dispatch_gpu <- log_lik_logvMEM(X, A_matrix, beta_true)

Sys.unsetenv("LOGVMEM_LIKELIHOOD_BACKEND")
Sys.unsetenv("LOGVMEM_N_QUAD")

cat("\nDispatcher check:\n")
cat("CPU dispatcher matches direct CPU:",
    isTRUE(all.equal(ll_dispatch_cpu, ll_cpu_ref, tolerance = 1e-12)), "\n")
cat("GPU dispatcher matches direct GPU:",
    isTRUE(all.equal(ll_dispatch_gpu, ll_gpu, tolerance = 1e-6)), "\n")

cat("\nTEST COMPLETE.\n")
