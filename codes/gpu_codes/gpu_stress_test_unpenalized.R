############################################################
# FULL UNPENALIZED MLE TEST
#
# Same dataset + same initial values + same optimizer
#
# Compare:
#
#   ORIGINAL adaptive CPU
#
#   CPU Gauss-Jacobi FP64:
#       Q = 64, 128, 256, 512
#
#   Apple MPS GPU FP32:
#       Q = 64, 128, 256, 512
#
############################################################

rm(list = ls())

############################################################
# 1. PROJECT DIRECTORY
############################################################

project_dir <- paste0(
  "/Users/rohan/Documents/Rohan/Finance paper/",
  "Sparse-Logarithmic-Vector-Multiplicative-Error-Model-",
  "with-Multivariate-Gamma-Errors-develop_RCodes"
)

setwd(project_dir)

############################################################
# 2. SOURCE TEST MLE FILE
############################################################

required_files <- c(
  "codes/gpu/multivariate_gamma_dist_gpu_test.R",
  "codes/gpu/mle_logvMEM_1_gpu_test.R",
  "codes/sim_logvMEM.R",
  "codes/generate_autoregressive_matrices.R"
)

missing_files <- required_files[
  !file.exists(required_files)
]

if (length(missing_files) > 0L) {
  stop(
    "Missing files: ",
    paste(missing_files, collapse = ", ")
  )
}

source(
  "codes/gpu/mle_logvMEM_1_gpu_test.R"
)

############################################################
# 3. GPU CHECK
############################################################

if (!requireNamespace("torch", quietly = TRUE)) {
  stop("Package 'torch' is not installed.")
}

mps_ok <- torch_mps_available_logvmem()

cat(
  "R architecture:",
  R.version$arch,
  "\n"
)

cat(
  "Machine:",
  Sys.info()[["machine"]],
  "\n"
)

cat(
  "Apple MPS available:",
  mps_ok,
  "\n"
)

if (!mps_ok) {
  stop("Apple MPS is not available.")
}

############################################################
# 4. SIMULATE EXACT SAME DATASET
############################################################

set.seed(12345)

d <- 5
p <- 10
q <- 0

N <- 1500
burn_in <- 500

beta_true <- 4

omega_true <- rep(
  0.1,
  d
)

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

A_matrix_list <-
  A_obj$A_matrix_list

A_true <- do.call(
  cbind,
  A_matrix_list
)

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

cat(
  "\nDataset dimensions:",
  paste(
    dim(X),
    collapse = " x "
  ),
  "\n"
)

############################################################
# 5. GENERATE INITIAL VALUES ONCE
############################################################
#
# CRITICAL:
#
# Every estimator below gets these EXACT SAME initial values.
#
############################################################

cat(
  "\nGenerating common initial values...\n"
)

initial_values <- init.val(
  X = X,
  maxlag = p,
  n_beta = 20
)

init_A_matrix <-
  initial_values$init_A_matrix

init_beta <-
  initial_values$beta_init

cat(
  "Initial beta:",
  init_beta,
  "\n"
)

cat(
  "Initial A dimensions:",
  paste(
    dim(init_A_matrix),
    collapse = " x "
  ),
  "\n"
)

############################################################
# 6. OPTIMIZATION SETTINGS
############################################################
#
# IMPORTANT:
# Set these to the SAME values you intend to use in the
# simulation study.
#
############################################################

max_outer_iter <- 20

max_block_sweeps <- 20

max_iter_bobyqa <- 200

lower_A <- -0.5
upper_A <- 1

beta_lower <- 0.2
beta_upper <- 20

eps_A <- 1e-3

tol_beta <- 1e-3

tol_outer_A <- 1e-3
tol_outer_beta <- 1e-3

############################################################
# 7. Q VALUES
############################################################

Q_grid <- c(
  64L,
  128L,
  256L,
  512L
)

############################################################
# 8. OUTPUT DIRECTORY
############################################################

output_dir <- file.path(
  paste0(project_dir,'codes/gpu/'),
  "gpu_unpenalized_validation"
)

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

############################################################
# 9. WARM UP GPU
############################################################

cat(
  "\nWarming up Apple MPS...\n"
)

invisible(
  log_lik_logvMEM_gpu(
    X = X[
      1:300,
      ,
      drop = FALSE
    ],
    A_matrix = A_true,
    beta_par = beta_true,
    n_quad = 64L,
    device = "mps"
  )
)

cat(
  "GPU warm-up complete.\n"
)

############################################################
# 10. HELPER TO RUN ONE FIT
############################################################

run_one_fit <- function(
    backend,
    Q = NA_integer_,
    label
) {
  
  cat(
    "\n\n============================================================\n"
  )
  
  cat(
    "STARTING:",
    label,
    "\n"
  )
  
  cat(
    "============================================================\n"
  )
  
  ##########################################################
  # Set likelihood backend
  ##########################################################
  
  Sys.setenv(
    LOGVMEM_LIKELIHOOD_BACKEND =
      backend
  )
  
  if (!is.na(Q)) {
    
    Sys.setenv(
      LOGVMEM_N_QUAD =
        as.character(Q)
    )
    
  } else {
    
    Sys.unsetenv(
      "LOGVMEM_N_QUAD"
    )
  }
  
  ##########################################################
  # Verify likelihood before optimizer starts
  ##########################################################
  
  initial_ll <- log_lik_logvMEM(
    X = X,
    A_matrix = init_A_matrix,
    beta_par = init_beta
  )
  
  cat(
    "Initial likelihood:",
    initial_ll,
    "\n"
  )
  
  ##########################################################
  # Full unpenalized fit
  ##########################################################
  
  timing <- system.time({
    
    fit <- estimate_method1_joint(
      
      X = X,
      
      # EXACT SAME initial values each time
      init_A_matrix =
        init_A_matrix,
      
      init_beta =
        init_beta,
      
      max_outer_iter =
        max_outer_iter,
      
      max_block_sweeps =
        max_block_sweeps,
      
      lower_A =
        lower_A,
      
      upper_A =
        upper_A,
      
      beta_lower =
        beta_lower,
      
      beta_upper =
        beta_upper,
      
      max_iter_bobyqa =
        max_iter_bobyqa,
      
      eps_A =
        eps_A,
      
      tol_beta =
        tol_beta,
      
      tol_outer_A =
        tol_outer_A,
      
      tol_outer_beta =
        tol_outer_beta,
      
      trace = TRUE
    )
    
  })
  
  ##########################################################
  # Re-evaluate final likelihood using SAME backend
  ##########################################################
  
  final_ll <- log_lik_logvMEM(
    
    X = X,
    
    A_matrix =
      fit$A_estimate,
    
    beta_par =
      fit$beta_estimate
  )
  
  ##########################################################
  # Save fit immediately
  ##########################################################
  
  fit_file <- file.path(
    
    output_dir,
    
    paste0(
      "fit_",
      label,
      ".rds"
    )
  )
  
  saveRDS(
    fit,
    fit_file
  )
  
  ##########################################################
  # Summary
  ##########################################################
  
  summary <- data.frame(
    
    label = label,
    
    backend = backend,
    
    Q = Q,
    
    beta_hat =
      fit$beta_estimate,
    
    loglik =
      final_ll,
    
    converged =
      fit$converged,
    
    n_outer_iter =
      fit$n_outer_iter,
    
    runtime_seconds =
      unname(
        timing[["elapsed"]]
      ),
    
    initial_loglik =
      initial_ll,
    
    stringsAsFactors = FALSE
  )
  
  cat(
    "\nFinished:",
    label,
    "\n"
  )
  
  cat(
    "beta_hat =",
    fit$beta_estimate,
    "\n"
  )
  
  cat(
    "loglik =",
    final_ll,
    "\n"
  )
  
  cat(
    "converged =",
    fit$converged,
    "\n"
  )
  
  cat(
    "outer iterations =",
    fit$n_outer_iter,
    "\n"
  )
  
  cat(
    "runtime seconds =",
    unname(
      timing[["elapsed"]]
    ),
    "\n"
  )
  
  list(
    fit = fit,
    summary = summary
  )
}

############################################################
# 11. ORIGINAL ADAPTIVE CPU FIT
############################################################

fit_results <- list()

fit_results[["cpu_reference"]] <- run_one_fit(
  
  backend =
    "cpu_reference",
  
  Q =
    NA_integer_,
  
  label =
    "cpu_reference"
)

############################################################
# 12. CPU GAUSS-JACOBI FITS
############################################################

for (Q in Q_grid) {
  
  label <- paste0(
    "cpu_jacobi_Q",
    Q
  )
  
  fit_results[[label]] <- run_one_fit(
    
    backend =
      "cpu_quadrature",
    
    Q =
      Q,
    
    label =
      label
  )
}

############################################################
# 13. GPU GAUSS-JACOBI FITS
############################################################

for (Q in Q_grid) {
  
  label <- paste0(
    "gpu_jacobi_Q",
    Q
  )
  
  fit_results[[label]] <- run_one_fit(
    
    backend =
      "gpu",
    
    Q =
      Q,
    
    label =
      label
  )
}

############################################################
# 14. CLEAN ENVIRONMENT VARIABLES
############################################################

Sys.unsetenv(
  "LOGVMEM_LIKELIHOOD_BACKEND"
)

Sys.unsetenv(
  "LOGVMEM_N_QUAD"
)

############################################################
# 15. BASIC SUMMARY TABLE
############################################################

summary_df <- do.call(
  
  rbind,
  
  lapply(
    fit_results,
    function(x) {
      x$summary
    }
  )
)

rownames(summary_df) <- NULL

############################################################
# 16. EXTRACT CPU REFERENCE ESTIMATE
############################################################

fit_ref <-
  fit_results[["cpu_reference"]]$fit

A_ref <-
  fit_ref$A_estimate

beta_ref <-
  fit_ref$beta_estimate

ll_ref <-
  fit_ref$loglik

############################################################
# 17. ADD DIFFERENCES RELATIVE TO CPU REFERENCE
############################################################

summary_df$beta_abs_diff_vs_reference <-
  NA_real_

summary_df$beta_rel_diff_vs_reference <-
  NA_real_

summary_df$A_frobenius_diff_vs_reference <-
  NA_real_

summary_df$A_max_abs_diff_vs_reference <-
  NA_real_

summary_df$A_rmse_vs_reference <-
  NA_real_

summary_df$loglik_abs_diff_vs_reference <-
  NA_real_

summary_df$loglik_rel_diff_vs_reference <-
  NA_real_

for (
  i in seq_len(
    nrow(summary_df)
  )
) {
  
  label <-
    summary_df$label[i]
  
  fit_i <-
    fit_results[[label]]$fit
  
  A_i <-
    fit_i$A_estimate
  
  beta_i <-
    fit_i$beta_estimate
  
  ll_i <-
    fit_i$loglik
  
  ##########################################################
  # beta
  ##########################################################
  
  summary_df$
    beta_abs_diff_vs_reference[i] <-
    abs(
      beta_i -
        beta_ref
    )
  
  summary_df$
    beta_rel_diff_vs_reference[i] <-
    abs(
      beta_i -
        beta_ref
    ) /
    max(
      1,
      abs(beta_ref)
    )
  
  ##########################################################
  # A
  ##########################################################
  
  A_diff <-
    A_i -
    A_ref
  
  summary_df$
    A_frobenius_diff_vs_reference[i] <-
    sqrt(
      sum(
        A_diff^2
      )
    )
  
  summary_df$
    A_max_abs_diff_vs_reference[i] <-
    max(
      abs(
        A_diff
      )
    )
  
  summary_df$
    A_rmse_vs_reference[i] <-
    sqrt(
      mean(
        A_diff^2
      )
    )
  
  ##########################################################
  # likelihood
  ##########################################################
  
  summary_df$
    loglik_abs_diff_vs_reference[i] <-
    abs(
      ll_i -
        ll_ref
    )
  
  summary_df$
    loglik_rel_diff_vs_reference[i] <-
    abs(
      ll_i -
        ll_ref
    ) /
    max(
      1,
      abs(ll_ref)
    )
}

############################################################
# 18. CPU-vs-GPU COMPARISON AT SAME Q
############################################################

same_Q_list <- list()

for (Q in Q_grid) {
  
  cpu_label <-
    paste0(
      "cpu_jacobi_Q",
      Q
    )
  
  gpu_label <-
    paste0(
      "gpu_jacobi_Q",
      Q
    )
  
  cpu_fit <-
    fit_results[[cpu_label]]$fit
  
  gpu_fit <-
    fit_results[[gpu_label]]$fit
  
  A_cpu <-
    cpu_fit$A_estimate
  
  A_gpu <-
    gpu_fit$A_estimate
  
  A_diff <-
    A_gpu -
    A_cpu
  
  same_Q_list[[as.character(Q)]] <- data.frame(
    
    Q = Q,
    
    beta_cpu =
      cpu_fit$beta_estimate,
    
    beta_gpu =
      gpu_fit$beta_estimate,
    
    beta_abs_diff =
      abs(
        cpu_fit$beta_estimate -
          gpu_fit$beta_estimate
      ),
    
    loglik_cpu =
      cpu_fit$loglik,
    
    loglik_gpu =
      gpu_fit$loglik,
    
    loglik_abs_diff =
      abs(
        cpu_fit$loglik -
          gpu_fit$loglik
      ),
    
    A_frobenius_diff =
      sqrt(
        sum(
          A_diff^2
        )
      ),
    
    A_max_abs_diff =
      max(
        abs(
          A_diff
        )
      ),
    
    A_rmse =
      sqrt(
        mean(
          A_diff^2
        )
      ),
    
    CPU_converged =
      cpu_fit$converged,
    
    GPU_converged =
      gpu_fit$converged,
    
    CPU_outer_iter =
      cpu_fit$n_outer_iter,
    
    GPU_outer_iter =
      gpu_fit$n_outer_iter,
    
    stringsAsFactors = FALSE
  )
}

same_Q_df <- do.call(
  rbind,
  same_Q_list
)

rownames(same_Q_df) <- NULL

############################################################
# 19. Q-TO-Q ESTIMATE STABILITY
############################################################
#
# Compare Q64 -> Q128 -> Q256 -> Q512 separately
# for CPU and GPU.
#
############################################################

Q_stability <- list()

stability_id <- 1L

for (
  backend_name in c(
    "cpu_jacobi",
    "gpu_jacobi"
  )
) {
  
  for (
    k in 2:length(Q_grid)
  ) {
    
    Q_old <-
      Q_grid[
        k - 1
      ]
    
    Q_new <-
      Q_grid[k]
    
    old_label <-
      paste0(
        backend_name,
        "_Q",
        Q_old
      )
    
    new_label <-
      paste0(
        backend_name,
        "_Q",
        Q_new
      )
    
    old_fit <-
      fit_results[[old_label]]$fit
    
    new_fit <-
      fit_results[[new_label]]$fit
    
    A_diff <-
      new_fit$A_estimate -
      old_fit$A_estimate
    
    Q_stability[[stability_id]] <- data.frame(
      
      backend =
        backend_name,
      
      Q_from =
        Q_old,
      
      Q_to =
        Q_new,
      
      beta_abs_change =
        abs(
          new_fit$beta_estimate -
            old_fit$beta_estimate
        ),
      
      loglik_abs_change =
        abs(
          new_fit$loglik -
            old_fit$loglik
        ),
      
      A_frobenius_change =
        sqrt(
          sum(
            A_diff^2
          )
        ),
      
      A_max_abs_change =
        max(
          abs(
            A_diff
          )
        ),
      
      A_rmse_change =
        sqrt(
          mean(
            A_diff^2
          )
        ),
      
      stringsAsFactors = FALSE
    )
    
    stability_id <-
      stability_id + 1L
  }
}

Q_stability_df <- do.call(
  rbind,
  Q_stability
)

############################################################
# 20. SPEED TABLE
############################################################

reference_runtime <-
  summary_df$
  runtime_seconds[
    summary_df$label ==
      "cpu_reference"
  ]

summary_df$speedup_vs_adaptive_cpu <-
  reference_runtime /
  summary_df$runtime_seconds

############################################################
# 21. PRINT RESULTS
############################################################

cat(
  "\n\n============================================================\n"
)

cat(
  "FULL FIT SUMMARY\n"
)

cat(
  "============================================================\n"
)

print(
  summary_df,
  row.names = FALSE,
  digits = 10
)

cat(
  "\n============================================================\n"
)

cat(
  "CPU VS GPU AT SAME Q\n"
)

cat(
  "============================================================\n"
)

print(
  same_Q_df,
  row.names = FALSE,
  digits = 10
)

cat(
  "\n============================================================\n"
)

cat(
  "ESTIMATE STABILITY AS Q INCREASES\n"
)

cat(
  "============================================================\n"
)

print(
  Q_stability_df,
  row.names = FALSE,
  digits = 10
)

############################################################
# 22. ALSO COMPARE ESTIMATES TO TRUE PARAMETERS
############################################################

truth_comparison <- list()

for (
  i in seq_len(
    nrow(summary_df)
  )
) {
  
  label <-
    summary_df$label[i]
  
  fit_i <-
    fit_results[[label]]$fit
  
  A_error <-
    fit_i$A_estimate -
    A_true
  
  truth_comparison[[i]] <- data.frame(
    
    label = label,
    
    beta_hat =
      fit_i$beta_estimate,
    
    beta_true =
      beta_true,
    
    beta_abs_error =
      abs(
        fit_i$beta_estimate -
          beta_true
      ),
    
    A_RMSE_vs_true =
      sqrt(
        mean(
          A_error^2
        )
      ),
    
    A_max_abs_error_vs_true =
      max(
        abs(
          A_error
        )
      ),
    
    stringsAsFactors = FALSE
  )
}

truth_df <- do.call(
  rbind,
  truth_comparison
)

cat(
  "\n============================================================\n"
)

cat(
  "ESTIMATES VS TRUE SIMULATION PARAMETERS\n"
)

cat(
  "============================================================\n"
)

print(
  truth_df,
  row.names = FALSE,
  digits = 10
)

############################################################
# 23. SAVE EVERYTHING
############################################################

write.csv(
  
  summary_df,
  
  file.path(
    output_dir,
    "unpenalized_CPU_GPU_Q_summary.csv"
  ),
  
  row.names = FALSE
)

write.csv(
  
  same_Q_df,
  
  file.path(
    output_dir,
    "unpenalized_CPU_vs_GPU_same_Q.csv"
  ),
  
  row.names = FALSE
)

write.csv(
  
  Q_stability_df,
  
  file.path(
    output_dir,
    "unpenalized_Q_stability.csv"
  ),
  
  row.names = FALSE
)

write.csv(
  
  truth_df,
  
  file.path(
    output_dir,
    "unpenalized_estimates_vs_truth.csv"
  ),
  
  row.names = FALSE
)

############################################################
# 24. SAVE COMMON INPUTS
############################################################

saveRDS(
  
  list(
    X = X,
    A_true = A_true,
    beta_true = beta_true,
    init_A_matrix =
      init_A_matrix,
    init_beta =
      init_beta,
    optimization_settings =
      list(
        max_outer_iter =
          max_outer_iter,
        max_block_sweeps =
          max_block_sweeps,
        max_iter_bobyqa =
          max_iter_bobyqa,
        eps_A =
          eps_A,
        tol_beta =
          tol_beta,
        tol_outer_A =
          tol_outer_A,
        tol_outer_beta =
          tol_outer_beta
      )
  ),
  
  file.path(
    output_dir,
    "common_test_inputs.rds"
  )
)

cat(
  "\n============================================================\n"
)

cat(
  "TEST COMPLETE\n"
)

cat(
  "Results saved in:\n",
  output_dir,
  "\n"
)

cat(
  "============================================================\n"
)