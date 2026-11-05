############################################################
# RESIDENT-DATA GPU BENCHMARK FOR CORRECTED MVGAMMA
#
# Compare at fixed T = 1500 and Q = 64:
#   1. CPU Jacobi FP64 ordinary call
#   2. GPU Jacobi FP32 ordinary end-to-end call
#   3. GPU Jacobi FP32 resident-data kernel
#
# Dimensions: d = 5, 50, 100, 200, 500
############################################################

rm(list = ls())

project_dir <- paste0(
  "/Users/rohan/Documents/Rohan/Finance paper/",
  "Sparse-Logarithmic-Vector-Multiplicative-Error-Model-",
  "with-Multivariate-Gamma-Errors-develop_RCodes/"
)
setwd(project_dir)

source("codes/gpu/multivariate_gamma_dist_gpu_test.R")

if (!requireNamespace("torch", quietly = TRUE)) stop("Package 'torch' is required.")
if (!torch_mps_available_logvmem()) stop("Apple MPS GPU is not available.")

cat("R architecture:", R.version$arch, "\n")
cat("Machine:", Sys.info()[["machine"]], "\n")
cat("torch:", as.character(packageVersion("torch")), "\n")
cat("Apple MPS available: TRUE\n")

# Settings
set.seed(12345)
T_fixed <- 1500L
d_grid <- c(5L, 50L, 100L, 200L, 500L)
beta_par <- 4
alpha <- beta_par / 2
Q <- 64L
n_timing_rep <- 5L
n_resident_eval <- 20L
batch_size <- 128L

prepare_mvgamma_mps_context <- function(x_mat, beta_par, Q = 64L, batch_size = 128L) {
  x_mat <- as.matrix(x_mat)
  storage.mode(x_mat) <- "double"
  
  n <- nrow(x_mat)
  d <- ncol(x_mat)
  
  alpha <- beta_par / 2
  lambda_vec <- rep(beta_par / 2, d)
  theta_vec <- rep(beta_par, d)
  
  if (any(!is.finite(x_mat)) || any(x_mat <= 0)) {
    stop("x_mat must contain finite positive values.")
  }
  
  B <- apply(x_mat, 1L, min)
  min_index <- max.col(-x_mat, ties.method = "first")
  
  min_mask <- matrix(0, nrow = n, ncol = d)
  min_mask[cbind(seq_len(n), min_index)] <- 1
  
  lambda_min <- beta_par / 2
  
  rule <- gauss_jacobi_rule_01(
    n = Q,
    a = lambda_min - 1,
    b = alpha - 1
  )
  
  log_const <- (
    alpha * log(beta_par) -
      lgamma(alpha) +
      sum(lambda_vec * log(theta_vec) - lgamma(lambda_vec))
  )
  
  dtype <- torch::torch_float32()
  
  list(
    n = n,
    d = d,
    Q = Q,
    batch_size = as.integer(batch_size),
    beta_par = beta_par,
    alpha = alpha,
    lambda_min = lambda_min,
    log_const = log_const,
    
    X = torch::torch_tensor(x_mat, dtype = dtype, device = "mps"),
    B = torch::torch_tensor(B, dtype = dtype, device = "mps"),
    min_mask = torch::torch_tensor(min_mask, dtype = dtype, device = "mps"),
    lambda = torch::torch_tensor(lambda_vec, dtype = dtype, device = "mps")$view(c(1, 1, d)),
    theta = torch::torch_tensor(theta_vec, dtype = dtype, device = "mps")$view(c(1, 1, d)),
    u = torch::torch_tensor(rule$nodes, dtype = dtype, device = "mps")$view(c(1, Q)),
    log_w = torch::torch_tensor(log(rule$weights), dtype = dtype, device = "mps")$view(c(1, Q))
  )
}

mvgamma_mps_resident_kernel <- function(ctx) {
  n <- ctx$n
  bs <- ctx$batch_size
  total_ll <- 0
  
  starts <- seq.int(1L, n, by = bs)
  
  for (start in starts) {
    end <- min(n, start + bs - 1L)
    ids <- start:end
    
    Xb <- ctx$X[ids, ]
    Bb <- ctx$B[ids]$unsqueeze(2)
    mask3 <- ctx$min_mask[ids, ]$unsqueeze(2)
    
    z <- Bb * ctx$u
    
    shifted <- Xb$unsqueeze(2) - z$unsqueeze(3)
    shifted <- torch::torch_clamp(shifted, min = 1e-30)
    
    log_shifted <- torch::torch_log(shifted)
    power_term <- (ctx$lambda - 1) * log_shifted
    
    log_component <- (
      power_term -
        ctx$theta * shifted -
        power_term * mask3
    )
    
    log_smooth <- torch::torch_sum(log_component, dim = 3) - ctx$beta_par * z
    log_terms <- log_smooth + ctx$log_w
    
    log_integral_core <- torch::torch_logsumexp(log_terms, dim = 2)
    
    log_B_factor <- (
      ctx$alpha + ctx$lambda_min - 1
    ) * torch::torch_log(Bb$squeeze(2))
    
    log_pdf <- log_integral_core + log_B_factor + ctx$log_const
    
    batch_sum <- torch::torch_sum(log_pdf)
    total_ll <- total_ll + as.numeric(batch_sum$to(device = "cpu"))
  }
  
  total_ll
}

cat("\nGlobal MPS warm-up...\n")
warm_x <- simulate_mvgamma(
  N = 300,
  m = 5,
  alpha = alpha,
  beta = beta_par,
  lambda_vec = rep(beta_par / 2, 5),
  theta_vec = rep(beta_par, 5)
)
warm_ctx <- prepare_mvgamma_mps_context(warm_x, beta_par, Q, batch_size)
invisible(mvgamma_mps_resident_kernel(warm_ctx))
rm(warm_ctx, warm_x)
gc()
cat("Warm-up complete.\n")

results <- vector("list", length(d_grid))

for (ii in seq_along(d_grid)) {
  d_now <- d_grid[ii]
  
  cat("\n============================================================\n")
  cat("d =", d_now, "| T =", T_fixed, "| Q =", Q, "\n")
  cat("============================================================\n")
  
  lambda_vec <- rep(beta_par / 2, d_now)
  theta_vec <- rep(beta_par, d_now)
  
  set.seed(20000 + d_now)
  
  X_test <- simulate_mvgamma(
    N = T_fixed,
    m = d_now,
    alpha = alpha,
    beta = beta_par,
    lambda_vec = lambda_vec,
    theta_vec = theta_vec
  )
  X_test <- as.matrix(X_test)
  
  # A. CPU ordinary call
  cpu_times <- numeric(n_timing_rep)
  ll_cpu <- NA_real_
  
  for (rr in seq_len(n_timing_rep)) {
    tm <- system.time({
      dens_cpu <- mvgamma_logpdf_batch_cpu_quadrature(
        x_mat = X_test,
        alpha = alpha,
        beta = beta_par,
        lambda_vec = lambda_vec,
        theta_vec = theta_vec,
        n_quad = Q
      )
      ll_cpu <- sum(dens_cpu)
    })
    cpu_times[rr] <- unname(tm[["elapsed"]])
  }
  
  # B. GPU ordinary end-to-end call
  gpu_full_times <- numeric(n_timing_rep)
  ll_gpu_full <- NA_real_
  
  for (rr in seq_len(n_timing_rep)) {
    tm <- system.time({
      dens_gpu <- mvgamma_logpdf_batch_gpu(
        x_mat = X_test,
        alpha = alpha,
        beta = beta_par,
        lambda_vec = lambda_vec,
        theta_vec = theta_vec,
        n_quad = Q,
        device = "mps"
      )
      ll_gpu_full <- sum(dens_gpu)
    })
    gpu_full_times[rr] <- unname(tm[["elapsed"]])
  }
  
  # C. Prepare resident GPU context ONCE
  prep_time <- system.time({
    ctx <- prepare_mvgamma_mps_context(
      x_mat = X_test,
      beta_par = beta_par,
      Q = Q,
      batch_size = batch_size
    )
  })
  prep_seconds <- unname(prep_time[["elapsed"]])
  
  invisible(mvgamma_mps_resident_kernel(ctx))
  
  # D. Resident GPU repeated kernel
  resident_block_times <- numeric(n_timing_rep)
  ll_gpu_resident <- NA_real_
  
  for (rr in seq_len(n_timing_rep)) {
    tm <- system.time({
      for (kk in seq_len(n_resident_eval)) {
        ll_gpu_resident <- mvgamma_mps_resident_kernel(ctx)
      }
    })
    resident_block_times[rr] <- unname(tm[["elapsed"]])
  }
  
  resident_per_eval <- resident_block_times / n_resident_eval
  
  cpu_sec <- median(cpu_times)
  gpu_full_sec <- median(gpu_full_times)
  gpu_resident_sec <- median(resident_per_eval)
  
  rel_full <- abs(ll_gpu_full - ll_cpu) / max(1, abs(ll_cpu))
  rel_resident <- abs(ll_gpu_resident - ll_cpu) / max(1, abs(ll_cpu))
  
  results[[ii]] <- data.frame(
    Sample_Size_T = T_fixed,
    Dimension_d = d_now,
    Q = Q,
    CPU_seconds = cpu_sec,
    GPU_end_to_end_seconds = gpu_full_sec,
    GPU_resident_seconds = gpu_resident_sec,
    GPU_context_setup_seconds = prep_seconds,
    CPU_over_GPU_end_to_end = cpu_sec / gpu_full_sec,
    CPU_over_GPU_resident = cpu_sec / gpu_resident_sec,
    Resident_over_end_to_end_improvement = gpu_full_sec / gpu_resident_sec,
    CPU_loglik = ll_cpu,
    GPU_end_to_end_loglik = ll_gpu_full,
    GPU_resident_loglik = ll_gpu_resident,
    GPU_end_to_end_relative_error = rel_full,
    GPU_resident_relative_error = rel_resident,
    stringsAsFactors = FALSE
  )
  
  cat("CPU ordinary            :", round(cpu_sec, 6), "sec/eval\n")
  cat("GPU ordinary end-to-end :", round(gpu_full_sec, 6), "sec/eval\n")
  cat("GPU resident kernel     :", round(gpu_resident_sec, 6), "sec/eval\n")
  cat("One-time GPU setup      :", round(prep_seconds, 6), "sec\n")
  cat("CPU / resident GPU      :", round(cpu_sec / gpu_resident_sec, 2), "x\n")
  cat("Resident improvement vs current GPU:", round(gpu_full_sec / gpu_resident_sec, 2), "x\n")
  cat("Resident GPU rel. error :", format(rel_resident, scientific = TRUE, digits = 6), "\n")
  
  rm(ctx, X_test)
  gc()
}

results_df <- do.call(rbind, results)
rownames(results_df) <- NULL

cat("\n\n============================================================\n")
cat("RESIDENT-DATA BENCHMARK RESULTS\n")
cat("============================================================\n")

print(
  results_df[, c(
    "Sample_Size_T",
    "Dimension_d",
    "Q",
    "CPU_seconds",
    "GPU_end_to_end_seconds",
    "GPU_resident_seconds",
    "GPU_context_setup_seconds",
    "CPU_over_GPU_end_to_end",
    "CPU_over_GPU_resident",
    "Resident_over_end_to_end_improvement",
    "GPU_resident_relative_error"
  )],
  row.names = FALSE,
  digits = 8
)

output_file <- file.path(
  paste0(project_dir,'codes/gpu'),
  paste0("mvgamma_resident_GPU_benchmark_T", T_fixed, "_Q", Q, ".csv")
)

write.csv(results_df, output_file, row.names = FALSE)

cat("\nSaved to:\n", output_file, "\n")
cat("\nTEST COMPLETE.\n")
