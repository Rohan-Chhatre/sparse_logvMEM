
############################################################
# REPEATED PENALIZED SIMULATION STUDY
#
# Adds:
#   1. B Monte Carlo replications
#   2. A master progress file (no console trace required)
#   3. BIC selection within each replication
#   4. Selection-frequency heatmaps across replications
#
# Important:
#   - The true A matrix is generated once per HLag structure.
#   - A new dataset is generated for every replication.
#   - Each heatmap cell equals the proportion of replications
#     in which that coefficient was selected as nonzero.
############################################################

rm(list = ls())

############################################################
# 1. PROJECT SETUP
############################################################

project_dir <- file.path(
  path.expand("~/Documents/Rohan/Finance paper"),
  paste0(
    "Sparse-Logarithmic-Vector-Multiplicative-Error-Model-",
    "with-Multivariate-Gamma-Errors-develop_RCodes"
  )
)

setwd(project_dir)

output_root <- file.path(
  project_dir,
  "simulation_study",
  "output_corrected_gamma_B"
)

dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

progress_file <- file.path(output_root, "MASTER_PROGRESS.txt")

writeLines(
  paste0(
    "Repeated penalized simulation study\n",
    "Started: ", Sys.time(), "\n"
  ),
  progress_file
)

log_progress <- function(...) {
  msg <- paste0(...)
  cat(
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    " | ",
    msg,
    "\n",
    file = progress_file,
    append = TRUE,
    sep = ""
  )
}

############################################################
# 2. SOURCE CURRENT FUNCTIONS
############################################################

source(file.path(project_dir, "codes", "mle_logvMEM_1.R"))
source(file.path(
  project_dir,
  "codes",
  "sparse_logvMEM_estimation_1_progress_bar.R"
))
source(file.path(project_dir, "codes", "sim_logvMEM.R"))
source(file.path(
  project_dir,
  "codes",
  "autoregressive_matrix_diag_dom.R"
))
source(file.path(
  project_dir,
  "codes",
  "generate_autoregressive_matrices.R"
))

mvgamma_file <- file.path(
  project_dir,
  "codes",
  "multivariate_gamma_dist.R"
)

if (file.exists(mvgamma_file)) {
  source(mvgamma_file)
}
############################################################
# PROGRESS-FILE HELPER
############################################################

write_progress_bar <- function(
    progress_file,
    i,
    total,
    status = NULL,
    ...
) {
  
  if (
    is.null(progress_file) ||
    length(progress_file) == 0 ||
    is.na(progress_file) ||
    !nzchar(progress_file)
  ) {
    return(invisible(NULL))
  }
  
  dir.create(
    dirname(progress_file),
    recursive = TRUE,
    showWarnings = FALSE
  )
  
  percentage <- if (total > 0) {
    100 * i / total
  } else {
    NA_real_
  }
  
  status_text <- if (
    !is.null(status) &&
    length(status) > 0 &&
    !is.na(status)
  ) {
    paste0(" | ", status)
  } else {
    ""
  }
  
  progress_text <- sprintf(
    "%s | %d/%d | %.1f%%%s\n",
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    i,
    total,
    percentage,
    status_text
  )
  
  cat(
    progress_text,
    file = progress_file,
    append = TRUE
  )
  
  invisible(NULL)
}
############################################################
# 3. PACKAGES
############################################################

required_packages <- c(
  "dplyr",
  "ACDm",
  "nloptr",
  "Rsolnp",
  "lattice"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

############################################################
# 4. SIMULATION SETTINGS
############################################################

# Set B = 1 to reproduce the original single-dataset experiment.
# Increase only after confirming that one replication completes.
B <- 100

d <- 5
p <- 10
q <- 0

N_train <- 1500
N_test <- 1
N_total <- N_train + N_test

burn_in <- 500

beta_true <- 4
omega_true <- rep(0.1, d)

simulation_seed <- 12345

############################################################
# 5. OPTIMIZATION SETTINGS
############################################################

threshold <- 1e-2

max_outer_iter <- 10
max_block_sweeps <- 1
max_iter_bobyqa <- 50

eps_A <- 1e-1
tol_beta <- 1e-1
tol_outer_A <- 1e-1
tol_outer_beta <- 1e-1

lower_A <- -0.5
upper_A <- 1

beta_lower <- 0.2
beta_upper <- 20

############################################################
# 6. LAMBDA GRID
############################################################

v <- 0.1
epsilon_lambda <- 1e-2
K_lambda <- 8

############################################################
# 7. PENALTIES AND STRUCTURES
############################################################

all_penalties <- c(
  "group-lasso",
  "group-mcp",
  "group-scad"
)

all_structures <- c(
  "componentwise",
  "own-other",
  "elementwise"
)

############################################################
# 8. MAX-LAG MATRICES
############################################################

get_maxlag_matrix <- function(Hlag_structure) {
  
  if (Hlag_structure == "componentwise") {
    
    matrix(
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
    
  } else if (Hlag_structure == "own-other") {
    
    matrix(
      c(
        2, 2, 2, 2, 2,
        2, 3, 2, 2, 2,
        3, 3, 4, 3, 3,
        3, 3, 3, 4, 3,
        6, 6, 6, 6, 7
      ),
      nrow = d,
      ncol = d,
      byrow = TRUE
    )
    
  } else if (Hlag_structure == "elementwise") {
    
    matrix(
      c(
        3, 0, 0, 3, 4,
        2, 4, 2, 2, 4,
        3, 3, 5, 4, 5,
        2, 2, 3, 5, 3,
        4, 4, 4, 5, 5
      ),
      nrow = d,
      ncol = d,
      byrow = TRUE
    )
    
  } else {
    stop("Unknown Hlag structure: ", Hlag_structure)
  }
}

############################################################
# 9. TRUE A MATRIX
############################################################

generate_true_A <- function(
    Hlag_structure,
    L,
    d,
    p,
    max_eigen_abs = 0.96
) {
  
  if (Hlag_structure == "own-other") {
    A_obj <- autoregressive_matrix_own_other_diag_dom(
      d = d,
      p = p,
      L = L,
      max_eigen_abs = max_eigen_abs
    )
  } else {
    A_obj <- autoregressive_matrix(
      d = d,
      p = p,
      L = L,
      max_eigen_abs = max_eigen_abs
    )
  }
  
  if (is.null(A_obj$A_matrix_list)) {
    stop("The A generator did not return A_matrix_list.")
  }
  
  A_matrix_full <- do.call(cbind, A_obj$A_matrix_list)
  
  if (!all(dim(A_matrix_full) == c(d, d * p))) {
    stop("Unexpected dimension of generated A matrix.")
  }
  
  list(
    A_matrix_list = A_obj$A_matrix_list,
    A_matrix_full = A_matrix_full
  )
}

############################################################
# 10. SUPPORT AND METRICS
############################################################

make_support_matrix <- function(A_matrix, threshold = 1e-3) {
  1L * (abs(A_matrix) > threshold)
}

safe_ratio <- function(x, y) {
  if (!is.finite(y) || y == 0) return(NA_real_)
  x / y
}

calculate_support_metrics <- function(
    A_estimated,
    A_true,
    threshold = 1e-3
) {
  
  estimated_active <- abs(A_estimated) > threshold
  true_active <- abs(A_true) > threshold
  
  TP <- sum(estimated_active & true_active)
  TN <- sum(!estimated_active & !true_active)
  FP <- sum(estimated_active & !true_active)
  FN <- sum(!estimated_active & true_active)
  
  data.frame(
    TP = TP,
    TN = TN,
    FP = FP,
    FN = FN,
    TPR = safe_ratio(TP, TP + FN),
    FPR = safe_ratio(FP, FP + TN),
    Precision = safe_ratio(TP, TP + FP),
    F1 = safe_ratio(2 * TP, 2 * TP + FP + FN),
    SupportAccuracy = safe_ratio(TP + TN, TP + TN + FP + FN),
    ExactRecovery = as.integer(FP == 0 && FN == 0)
  )
}

############################################################
# 11. BINARY SPARSITY PLOT
############################################################

SparsityPlot <- function(Bmat, p, k, title = NULL) {
  
  lag_labels <- vector("expression", p)
  
  for (lag_id in seq_len(p)) {
    lag_labels[[lag_id]] <- bquote(
      bold(A)^"(" * .(lag_id) * ")"
    )
  }
  
  rotate_matrix <- function(x) {
    t(x)[, nrow(x):1, drop = FALSE]
  }
  
  lag_centres <- seq(
    from = k / 2 + 0.5,
    to = p * k + 0.5,
    by = k
  )
  
  lattice::levelplot(
    rotate_matrix(abs(Bmat)),
    col.regions = colorRampPalette(
      c("white", "grey"),
      space = "Lab"
    ),
    at = c(-0.5, 0.5, 1.5),
    colorkey = FALSE,
    xlab = NULL,
    ylab = NULL,
    main = list(label = title, cex = 1),
    panel = function(...) {
      lattice::panel.levelplot(...)
      lattice::panel.abline(
        h = seq(1.5, nrow(Bmat) + 0.5, by = 1),
        v = seq(1.5, ncol(Bmat) + 0.5, by = 1),
        lwd = 0.5
      )
      lattice::panel.abline(
        v = seq(k + 0.5, p * k - 0.5, by = k),
        lwd = 2
      )
    },
    scales = list(
      x = list(
        alternating = 1,
        labels = lag_labels,
        at = lag_centres,
        cex = 0.8,
        tck = c(0, 0)
      ),
      y = list(draw = FALSE, tck = c(0, 0))
    )
  )
}

############################################################
# 12. SELECTION-FREQUENCY HEATMAP
############################################################

SelectionFrequencyPlot <- function(
    frequency_matrix,
    p,
    k,
    title = NULL
) {
  
  if (any(!is.finite(frequency_matrix))) {
    stop("Selection-frequency matrix contains non-finite values.")
  }
  
  if (any(frequency_matrix < 0 | frequency_matrix > 1)) {
    stop("Selection frequencies must lie between 0 and 1.")
  }
  
  lag_labels <- vector("expression", p)
  
  for (lag_id in seq_len(p)) {
    lag_labels[[lag_id]] <- bquote(
      bold(A)^"(" * .(lag_id) * ")"
    )
  }
  
  rotate_matrix <- function(x) {
    t(x)[, nrow(x):1, drop = FALSE]
  }
  
  lag_centres <- seq(
    from = k / 2 + 0.5,
    to = p * k + 0.5,
    by = k
  )
  
  lattice::levelplot(
    rotate_matrix(frequency_matrix),
    at = seq(0, 1, length.out = 101),
    col.regions = colorRampPalette(
      c("white", "grey85", "grey50", "grey20", "black"),
      space = "Lab"
    ),
    colorkey = list(
      title = "Selection\nfrequency",
      labels = list(
        at = c(0, 0.25, 0.5, 0.75, 1),
        labels = c("0", "0.25", "0.50", "0.75", "1")
      )
    ),
    xlab = NULL,
    ylab = NULL,
    main = list(label = title, cex = 1),
    panel = function(...) {
      lattice::panel.levelplot(...)
      lattice::panel.abline(
        h = seq(1.5, nrow(frequency_matrix) + 0.5, by = 1),
        v = seq(1.5, ncol(frequency_matrix) + 0.5, by = 1),
        lwd = 0.5
      )
      lattice::panel.abline(
        v = seq(k + 0.5, p * k - 0.5, by = k),
        lwd = 2
      )
    },
    scales = list(
      x = list(
        alternating = 1,
        labels = lag_labels,
        at = lag_centres,
        cex = 0.8,
        tck = c(0, 0)
      ),
      y = list(draw = FALSE, tck = c(0, 0))
    )
  )
}

############################################################
# 13. EXTRACT MINIMUM-BIC FIT
############################################################

extract_best_bic_fit <- function(
    sparse_result,
    penalty,
    Hlag_structure
) {
  
  if (is.null(sparse_result$summary_df)) {
    stop("run_sparse_grid_new() did not return summary_df.")
  }
  
  summary_df <- sparse_result$summary_df
  
  if (!"bic_full_data" %in% names(summary_df)) {
    stop("summary_df does not contain bic_full_data.")
  }
  
  valid_rows <- which(is.finite(summary_df$bic_full_data))
  
  if (length(valid_rows) == 0) {
    stop(
      "No finite BIC values for ",
      Hlag_structure,
      " / ",
      penalty
    )
  }
  
  best_row <- valid_rows[
    which.min(summary_df$bic_full_data[valid_rows])
  ]
  
  best_fit_object <- sparse_result$all_fits[[best_row]]
  
  if (
    is.list(best_fit_object) &&
    !is.null(best_fit_object$fit)
  ) {
    best_fit <- best_fit_object$fit
  } else {
    best_fit <- best_fit_object
  }
  
  if (is.null(best_fit$A_estimate)) {
    str(best_fit_object)
    stop("Could not find A_estimate in the selected fit.")
  }
  
  list(
    best_row = best_row,
    summary = summary_df[best_row, , drop = FALSE],
    fit = best_fit,
    all_fit_information = best_fit_object
  )
}

############################################################
# 14. RUN ONE REPLICATION
############################################################

run_one_replication <- function(
    Hlag_structure,
    replication_id,
    A_matrix_list_true,
    A_matrix_true,
    structure_dir,
    replication_seed
) {
  
  replication_dir <- file.path(
    structure_dir,
    sprintf("rep_%03d", replication_id)
  )
  
  dir.create(
    replication_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )
  
  completed_file <- file.path(replication_dir, "COMPLETED.rds")
  
  if (file.exists(completed_file)) {
    log_progress(
      Hlag_structure,
      " | replication ",
      replication_id,
      " already complete; reading saved result."
    )
    return(readRDS(completed_file))
  }
  
  log_progress(
    Hlag_structure,
    " | replication ",
    replication_id,
    "/",
    B,
    " started."
  )
  
  replication_start <- Sys.time()
  
  set.seed(replication_seed)
  
  X <- sim_logvMEM(
    N = N_total,
    d = d,
    omega = omega_true,
    A_matrix_list = A_matrix_list_true,
    B_matrix_list = NULL,
    beta_par = beta_true,
    burn_in = burn_in
  )
  
  X <- as.matrix(X)
  
  if (!all(dim(X) == c(N_total, d))) {
    stop("sim_logvMEM returned unexpected dimensions.")
  }
  
  X_train <- X[seq_len(N_train), , drop = FALSE]
  X_test <- X[N_total, , drop = FALSE]
  
  write.csv(
    X,
    file.path(replication_dir, "simulated_data.csv"),
    row.names = FALSE
  )
  
  log_progress(
    Hlag_structure,
    " | replication ",
    replication_id,
    " | initial values started."
  )
  
  initial_values <- init.val(
    X = X_train,
    maxlag = p,
    n_beta = 20
  )
  
  init_A_matrix <- initial_values$init_A_matrix
  init_beta <- initial_values$beta_init
  
  if (is.null(init_beta)) {
    if (!is.null(initial_values$beta_list)) {
      init_beta <- median(initial_values$beta_list, na.rm = TRUE)
    } else {
      init_beta <- 2
    }
  }
  
  unpenalized_rds <- file.path(
    replication_dir,
    "unpenalized_fit.rds"
  )
  
  if (file.exists(unpenalized_rds)) {
    
    fit_unpen <- readRDS(unpenalized_rds)
    
  } else {
    
    log_progress(
      Hlag_structure,
      " | replication ",
      replication_id,
      " | unpenalized fit started."
    )
    
    fit_unpen <- estimate_method1_joint(
      X = X_train,
      init_A_matrix = init_A_matrix,
      init_beta = init_beta,
      max_outer_iter = max_outer_iter,
      max_block_sweeps = max_block_sweeps,
      lower_A = lower_A,
      upper_A = upper_A,
      beta_lower = beta_lower,
      beta_upper = beta_upper,
      max_iter_bobyqa = max_iter_bobyqa,
      eps_A = eps_A,
      tol_beta = tol_beta,
      tol_outer_A = tol_outer_A,
      tol_outer_beta = tol_outer_beta,
      trace = TRUE
    )
    
    saveRDS(fit_unpen, unpenalized_rds)
    
    log_progress(
      Hlag_structure,
      " | replication ",
      replication_id,
      " | unpenalized fit completed; beta_hat = ",
      signif(fit_unpen$beta_estimate, 6),
      "."
    )
  }
  
  A_matrix_estimated_mle <- fit_unpen$A_estimate
  beta_mle <- fit_unpen$beta_estimate
  
  loglik_mle <- log_lik_logvMEM(
    X = X_train,
    A_matrix = A_matrix_estimated_mle,
    beta_par = beta_mle
  )
  
  lambda_max <- abs(loglik_mle) / (nrow(X_train) * p * v)
  
  if (!is.finite(lambda_max) || lambda_max <= 0) {
    stop("Invalid lambda_max: ", lambda_max)
  }
  
  lambda_grid <- round(
    exp(
      seq(
        log(lambda_max),
        log(lambda_max * epsilon_lambda),
        length.out = K_lambda
      )
    ),
    digits = 10
  )
  
  write.csv(
    data.frame(lambda = lambda_grid),
    file.path(replication_dir, "lambda_grid.csv"),
    row.names = FALSE
  )
  
  penalty_results <- vector("list", length(all_penalties))
  names(penalty_results) <- all_penalties
  
  metric_results <- vector("list", length(all_penalties))
  names(metric_results) <- all_penalties
  
  for (penalty in all_penalties) {
    
    log_progress(
      Hlag_structure,
      " | replication ",
      replication_id,
      " | ",
      penalty,
      " grid started."
    )
    
    penalty_rds <- file.path(
      replication_dir,
      paste0("penalized_grid_", penalty, ".rds")
    )
    
    if (file.exists(penalty_rds)) {
      
      sparse_result <- readRDS(penalty_rds)
      
    } else {
      
      # trace = FALSE suppresses console output.
      # progress_file writes internal progress if the current
      # run_sparse_grid_new() implementation supports it.
      internal_progress_file <- file.path(
        replication_dir,
        paste0("internal_progress_", penalty, ".txt")
      )
      
      sparse_result <- run_sparse_grid_new(
        X_train = X_train,
        X_test = X_test,
        h_step = 1,
        lambda_grid = lambda_grid,
        penalty_vec = penalty,
        Hlag_vec = Hlag_structure,
        maxlag = p,
        n_beta = 20,
        threshold = threshold,
        max_outer_iter = max_outer_iter,
        max_block_sweeps = max_block_sweeps,
        max_iter_bobyqa = max_iter_bobyqa,
        eps_A = eps_A,
        tol_beta = tol_beta,
        tol_outer_A = tol_outer_A,
        tol_outer_beta = tol_outer_beta,
        init_A_matrix = init_A_matrix,
        init_beta = init_beta,
        mle_A_matrix = A_matrix_estimated_mle,
        fit_unpen = fit_unpen,
        start_from_mle = TRUE,
        trace = TRUE,
        progress_file = internal_progress_file
      )
      
      saveRDS(sparse_result, penalty_rds)
    }
    
    write.csv(
      sparse_result$summary_df,
      file.path(
        replication_dir,
        paste0("all_lambda_results_", penalty, ".csv")
      ),
      row.names = FALSE
    )
    
    best_result <- extract_best_bic_fit(
      sparse_result = sparse_result,
      penalty = penalty,
      Hlag_structure = Hlag_structure
    )
    
    A_estimated <- as.matrix(best_result$fit$A_estimate)
    support <- make_support_matrix(A_estimated, threshold)
    
    write.csv(
      A_estimated,
      file.path(
        replication_dir,
        paste0("A_estimated_BIC_", penalty, ".csv")
      ),
      row.names = FALSE
    )
    
    write.csv(
      support,
      file.path(
        replication_dir,
        paste0("support_BIC_", penalty, ".csv")
      ),
      row.names = FALSE
    )
    
    metrics <- calculate_support_metrics(
      A_estimated = A_estimated,
      A_true = A_matrix_true,
      threshold = threshold
    )
    
    metrics$Hlag_structure <- Hlag_structure
    metrics$replication <- replication_id
    metrics$penalty <- penalty
    metrics$lambda <- if (
      "lambda" %in% names(best_result$summary)
    ) {
      best_result$summary$lambda
    } else {
      NA_real_
    }
    metrics$bic_full_data <- best_result$summary$bic_full_data
    
    metric_results[[penalty]] <- metrics
    
    penalty_results[[penalty]] <- list(
      A_estimated = A_estimated,
      support = support,
      best_summary = best_result$summary
    )
    
    log_progress(
      Hlag_structure,
      " | replication ",
      replication_id,
      " | ",
      penalty,
      " completed."
    )
  }
  
  replication_end <- Sys.time()
  
  result <- list(
    replication = replication_id,
    seed = replication_seed,
    beta_estimated_unpenalized = beta_mle,
    unpenalized_converged = fit_unpen$converged,
    penalty_results = penalty_results,
    metrics = do.call(rbind, metric_results),
    runtime_seconds = as.numeric(
      difftime(replication_end, replication_start, units = "secs")
    )
  )
  
  saveRDS(result, completed_file)
  
  log_progress(
    Hlag_structure,
    " | replication ",
    replication_id,
    "/",
    B,
    " completed in ",
    round(result$runtime_seconds / 60, 2),
    " minutes."
  )
  
  result
}

############################################################
# 15. RUN ONE STRUCTURE FOR B REPLICATIONS
############################################################

run_one_structure <- function(
    Hlag_structure,
    structure_seed
) {
  
  structure_dir <- file.path(output_root, Hlag_structure)
  
  dir.create(
    structure_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )
  
  log_progress(
    "Structure ",
    Hlag_structure,
    " started with B = ",
    B,
    "."
  )
  
  # The data-generating coefficient matrix remains fixed
  # across all B replications.
  set.seed(structure_seed)
  
  L <- get_maxlag_matrix(Hlag_structure)
  
  A_true_obj <- generate_true_A(
    Hlag_structure = Hlag_structure,
    L = L,
    d = d,
    p = p,
    max_eigen_abs = 0.96
  )
  
  A_matrix_list_true <- A_true_obj$A_matrix_list
  A_matrix_true <- A_true_obj$A_matrix_full
  true_support <- make_support_matrix(A_matrix_true, threshold)
  
  write.csv(
    L,
    file.path(structure_dir, "maxlag_matrix.csv"),
    row.names = FALSE
  )
  
  write.csv(
    A_matrix_true,
    file.path(structure_dir, "A_matrix_true.csv"),
    row.names = FALSE
  )
  
  write.csv(
    true_support,
    file.path(structure_dir, "true_support.csv"),
    row.names = FALSE
  )
  
  pdf(
    file.path(structure_dir, "true_sparsity.pdf"),
    width = 14,
    height = 2.7
  )
  print(
    SparsityPlot(
      Bmat = true_support,
      p = p,
      k = d,
      title = paste("True support:", Hlag_structure)
    )
  )
  dev.off()
  
  replication_results <- vector("list", B)
  
  for (b in seq_len(B)) {
    
    replication_seed <- (
      simulation_seed +
        100000L * match(Hlag_structure, all_structures) +
        b
    )
    
    replication_results[[b]] <- run_one_replication(
      Hlag_structure = Hlag_structure,
      replication_id = b,
      A_matrix_list_true = A_matrix_list_true,
      A_matrix_true = A_matrix_true,
      structure_dir = structure_dir,
      replication_seed = replication_seed
    )
    
    # Save after every replication.
    saveRDS(
      replication_results,
      file.path(structure_dir, "replication_results_partial.rds")
    )
  }
  
  ##########################################################
  # Aggregate selection frequencies
  ############################################################
  
  frequency_matrices <- vector("list", length(all_penalties))
  names(frequency_matrices) <- all_penalties
  
  for (penalty in all_penalties) {
    
    support_sum <- matrix(
      0,
      nrow = d,
      ncol = d * p
    )
    
    for (b in seq_len(B)) {
      support_sum <- support_sum +
        replication_results[[b]]$
        penalty_results[[penalty]]$
        support
    }
    
    selection_frequency <- support_sum / B
    
    frequency_matrices[[penalty]] <- selection_frequency
    
    write.csv(
      selection_frequency,
      file.path(
        structure_dir,
        paste0("selection_frequency_", penalty, ".csv")
      ),
      row.names = FALSE
    )
    
    pdf(
      file.path(
        structure_dir,
        paste0("selection_frequency_heatmap_", penalty, ".pdf")
      ),
      width = 14,
      height = 3.2
    )
    
    print(
      SelectionFrequencyPlot(
        frequency_matrix = selection_frequency,
        p = p,
        k = d,
        title = paste0(
          Hlag_structure,
          " — ",
          penalty,
          " (B = ",
          B,
          ")"
        )
      )
    )
    
    dev.off()
  }
  
  all_metrics <- do.call(
    rbind,
    lapply(replication_results, function(x) x$metrics)
  )
  
  rownames(all_metrics) <- NULL
  
  write.csv(
    all_metrics,
    file.path(structure_dir, "support_metrics_all_replications.csv"),
    row.names = FALSE
  )
  
  metric_summary <- all_metrics |>
    dplyr::group_by(Hlag_structure, penalty) |>
    dplyr::summarise(
      B_completed = dplyr::n(),
      mean_TPR = mean(TPR, na.rm = TRUE),
      mean_FPR = mean(FPR, na.rm = TRUE),
      mean_Precision = mean(Precision, na.rm = TRUE),
      mean_F1 = mean(F1, na.rm = TRUE),
      mean_SupportAccuracy = mean(
        SupportAccuracy,
        na.rm = TRUE
      ),
      exact_recovery_rate = mean(
        ExactRecovery,
        na.rm = TRUE
      ),
      .groups = "drop"
    )
  
  write.csv(
    metric_summary,
    file.path(structure_dir, "support_metrics_summary.csv"),
    row.names = FALSE
  )
  
  saveRDS(
    list(
      structure = Hlag_structure,
      L = L,
      A_true = A_matrix_true,
      true_support = true_support,
      replication_results = replication_results,
      selection_frequencies = frequency_matrices,
      metrics = all_metrics,
      metric_summary = metric_summary
    ),
    file.path(structure_dir, "structure_complete.rds")
  )
  
  log_progress(
    "Structure ",
    Hlag_structure,
    " completed."
  )
  
  list(
    structure = Hlag_structure,
    A_true = A_matrix_true,
    true_support = true_support,
    replication_results = replication_results,
    selection_frequencies = frequency_matrices,
    metrics = all_metrics,
    metric_summary = metric_summary
  )
}

############################################################
# 16. RUN ALL STRUCTURES
############################################################

study_start_time <- Sys.time()

log_progress(
  "Study settings: B = ",
  B,
  "; K_lambda = ",
  K_lambda,
  "; structures = ",
  paste(all_structures, collapse = ", "),
  "; penalties = ",
  paste(all_penalties, collapse = ", "),
  "."
)

all_results <- vector("list", length(all_structures))
names(all_results) <- all_structures

for (structure_id in seq_along(all_structures)) {
  
  Hlag_structure <- all_structures[structure_id]
  structure_seed <- simulation_seed + structure_id - 1
  
  all_results[[Hlag_structure]] <- run_one_structure(
    Hlag_structure = Hlag_structure,
    structure_seed = structure_seed
  )
  
  saveRDS(
    all_results,
    file.path(output_root, "all_results_partial.rds")
  )
}

############################################################
# 17. COMBINE METRIC SUMMARIES
############################################################

all_metric_summaries <- do.call(
  rbind,
  lapply(all_results, function(x) x$metric_summary)
)

rownames(all_metric_summaries) <- NULL

write.csv(
  all_metric_summaries,
  file.path(output_root, "support_metrics_summary_all_structures.csv"),
  row.names = FALSE
)

saveRDS(
  all_results,
  file.path(output_root, "all_results_complete.rds")
)

study_end_time <- Sys.time()

log_progress(
  "Entire study completed in ",
  round(
    as.numeric(
      difftime(study_end_time, study_start_time, units = "hours")
    ),
    3
  ),
  " hours."
)

############################################################
# 18. OPTIONAL COMBINED HEATMAP PDF
############################################################

combined_heatmap_file <- file.path(
  output_root,
  paste0("combined_selection_frequency_heatmaps_B", B, ".pdf")
)

pdf(
  combined_heatmap_file,
  width = 14,
  height = 3.2
)

for (Hlag_structure in all_structures) {
  for (penalty in all_penalties) {
    
    print(
      SelectionFrequencyPlot(
        frequency_matrix =
          all_results[[Hlag_structure]]$
          selection_frequencies[[penalty]],
        p = p,
        k = d,
        title = paste0(
          Hlag_structure,
          " — ",
          penalty,
          " (B = ",
          B,
          ")"
        )
      )
    )
  }
}

dev.off()

log_progress(
  "Combined heatmap PDF saved to: ",
  combined_heatmap_file
)

