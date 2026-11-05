############################################################
# UPDATED SINGLE-DATASET PENALIZED SIMULATION STUDY
#
# Structures:
#   1. componentwise
#   2. own-other
#   3. elementwise
#
# Estimation:
#   Corrected multivariate Gamma likelihood
#   BCD + BOBYQA
#
# Tuning:
#   BIC only
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
  "output_corrected_gamma"
)

dir.create(
  output_root,
  recursive = TRUE,
  showWarnings = FALSE
)

############################################################
# 2. SOURCE CURRENT FUNCTIONS
############################################################

source(file.path(
  project_dir,
  "codes",
  "mle_logvMEM_1.R"
))

source(file.path(
  project_dir,
  "codes",
  "sparse_logvMEM_estimation_1_progress_bar.R"
))

source(file.path(
  project_dir,
  "codes",
  "sim_logvMEM.R"
))

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

# Source this only when it is not already sourced inside the files above.
mvgamma_file <- file.path(
  project_dir,
  "codes",
  "multivariate_gamma_dist.R"
)

if (file.exists(mvgamma_file)) {
  source(mvgamma_file)
}

############################################################
# 3. LOAD PACKAGES
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
    install.packages(
      pkg,
      repos = "https://cloud.r-project.org"
    )
  }
  
  library(
    pkg,
    character.only = TRUE
  )
}

############################################################
# 4. GLOBAL SIMULATION SETTINGS
############################################################

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

threshold <- 1e-3

max_outer_iter <- 20
max_block_sweeps <- 1
max_iter_bobyqa <- 300

eps_A <- 1e-2
tol_beta <- 1e-2
tol_outer_A <- 1e-2
tol_outer_beta <- 1e-2

lower_A <- -0.5
upper_A <- 1

beta_lower <- 0.2
beta_upper <- 20

############################################################
# 6. LAMBDA GRID SETTINGS
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
    
  } else if (Hlag_structure == "own-other") {
    
    L <- matrix(
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
    
    L <- matrix(
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
    
    stop(
      "Unknown Hlag structure: ",
      Hlag_structure
    )
  }
  
  L
}

############################################################
# 9. CREATE TRUE AUTOREGRESSIVE MATRICES
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
    stop(
      "The autoregressive matrix function did not return ",
      "'A_matrix_list'."
    )
  }
  
  A_matrix_list <- A_obj$A_matrix_list
  A_matrix_full <- do.call(
    cbind,
    A_matrix_list
  )
  
  if (!all(dim(A_matrix_full) == c(d, d * p))) {
    stop(
      "The generated A matrix has dimension ",
      paste(dim(A_matrix_full), collapse = " x "),
      "; expected ",
      d,
      " x ",
      d * p,
      "."
    )
  }
  
  list(
    A_matrix_list = A_matrix_list,
    A_matrix_full = A_matrix_full
  )
}

############################################################
# 10. BINARY SUPPORT MATRIX
############################################################

make_support_matrix <- function(
    A_matrix,
    threshold = 1e-3
) {
  
  out <- matrix(
    0,
    nrow = nrow(A_matrix),
    ncol = ncol(A_matrix)
  )
  
  out[abs(A_matrix) > threshold] <- 1
  
  out
}

############################################################
# 11. SUPPORT-RECOVERY METRICS
############################################################

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
  
  safe_ratio <- function(numerator, denominator) {
    
    if (
      length(denominator) == 0 ||
      is.na(denominator) ||
      denominator == 0
    ) {
      return(NA_real_)
    }
    
    numerator / denominator
  }
  
  data.frame(
    TP = TP,
    TN = TN,
    FP = FP,
    FN = FN,
    TPR = safe_ratio(TP, TP + FN),
    FPR = safe_ratio(FP, FP + TN),
    Precision = safe_ratio(TP, TP + FP),
    F1 = safe_ratio(
      2 * TP,
      2 * TP + FP + FN
    ),
    SupportAccuracy = safe_ratio(
      TP + TN,
      TP + TN + FP + FN
    ),
    ExactRecovery = as.integer(
      FP == 0 && FN == 0
    ),
    stringsAsFactors = FALSE
  )
}

############################################################
# 12. SPARSITY PLOT FUNCTION
############################################################

SparsityPlot <- function(
    B,
    p,
    k,
    title = NULL
) {
  
  lag_labels <- vector(
    "expression",
    p
  )
  
  for (lag_id in seq_len(p)) {
    
    lag_labels[[lag_id]] <- bquote(
      bold(A)^"(" * .(lag_id) * ")"
    )
  }
  
  rotate_matrix <- function(x) {
    t(x)[, nrow(x):1, drop = FALSE]
  }
  
  palette_function <- colorRampPalette(
    c("white", "grey"),
    space = "Lab"
  )
  
  lag_centres <- seq(
    from = k / 2 + 0.5,
    to = p * k + 0.5,
    by = k
  )
  
  lattice::levelplot(
    rotate_matrix(abs(B)),
    col.regions = palette_function,
    at = c(-0.5, 0.5, 1.5),
    colorkey = FALSE,
    xlab = NULL,
    ylab = NULL,
    main = list(
      label = title,
      cex = 1
    ),
    panel = function(...) {
      
      lattice::panel.levelplot(...)
      
      lattice::panel.abline(
        h = seq(
          1.5,
          nrow(B) + 0.5,
          by = 1
        ),
        v = seq(
          1.5,
          ncol(B) + 0.5,
          by = 1
        ),
        lwd = 0.5
      )
      
      lag_boundaries <- seq(
        k + 0.5,
        p * k - 0.5,
        by = k
      )
      
      lattice::panel.abline(
        v = lag_boundaries,
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
      y = list(
        draw = FALSE,
        tck = c(0, 0)
      )
    )
  )
}

############################################################
# 13. EXTRACT FIT AT MINIMUM BIC
############################################################

extract_best_bic_fit <- function(
    sparse_result,
    penalty,
    Hlag_structure
) {
  
  if (is.null(sparse_result$summary_df)) {
    stop(
      "run_sparse_grid_new() did not return summary_df."
    )
  }
  
  summary_df <- sparse_result$summary_df
  
  if (!"bic_full_data" %in% names(summary_df)) {
    stop(
      "summary_df does not contain bic_full_data."
    )
  }
  
  valid_rows <- which(
    is.finite(summary_df$bic_full_data)
  )
  
  if (length(valid_rows) == 0) {
    stop(
      "No finite BIC values were returned for ",
      Hlag_structure,
      " / ",
      penalty,
      "."
    )
  }
  
  best_row <- valid_rows[
    which.min(
      summary_df$bic_full_data[valid_rows]
    )
  ]
  
  best_summary <- summary_df[
    best_row,
    ,
    drop = FALSE
  ]
  
  if (is.null(sparse_result$all_fits)) {
    stop(
      "run_sparse_grid_new() did not return all_fits."
    )
  }
  
  if (length(sparse_result$all_fits) < best_row) {
    stop(
      "The number of all_fits elements does not match ",
      "the number of summary rows."
    )
  }
  
  best_fit_object <- sparse_result$all_fits[[best_row]]
  
  # Current setup appears to store the fitted model under $fit.
  if (
    is.list(best_fit_object) &&
    !is.null(best_fit_object$fit)
  ) {
    best_fit <- best_fit_object$fit
  } else {
    best_fit <- best_fit_object
  }
  
  if (is.null(best_fit$A_estimate)) {
    
    cat(
      "\nStructure of selected all_fits element:\n"
    )
    
    print(
      str(best_fit_object)
    )
    
    stop(
      "Could not find A_estimate in the selected fit."
    )
  }
  
  list(
    best_row = best_row,
    summary = best_summary,
    fit = best_fit,
    all_fit_information = best_fit_object
  )
}

############################################################
# 14. RUN ONE HIERARCHICAL STRUCTURE
############################################################

run_one_structure <- function(
    Hlag_structure,
    structure_seed
) {
  
  cat(
    "\n====================================================\n"
  )
  
  cat(
    "Running structure:",
    Hlag_structure,
    "\n"
  )
  
  cat(
    "====================================================\n"
  )
  
  structure_dir <- file.path(
    output_root,
    Hlag_structure
  )
  
  dir.create(
    structure_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )
  
  structure_start_time <- Sys.time()
  
  ##########################################################
  # 14.1 Generate true A matrices
  ##########################################################
  
  set.seed(structure_seed)
  
  L <- get_maxlag_matrix(
    Hlag_structure
  )
  
  A_true_obj <- generate_true_A(
    Hlag_structure = Hlag_structure,
    L = L,
    d = d,
    p = p,
    max_eigen_abs = 0.96
  )
  
  A_matrix_list_true <- A_true_obj$A_matrix_list
  A_matrix_true <- A_true_obj$A_matrix_full
  
  write.csv(
    L,
    file.path(
      structure_dir,
      paste0(
        "maxlag_matrix_",
        Hlag_structure,
        ".csv"
      )
    ),
    row.names = FALSE
  )
  
  write.csv(
    A_matrix_true,
    file.path(
      structure_dir,
      paste0(
        "A_matrix_true_",
        Hlag_structure,
        ".csv"
      )
    ),
    row.names = FALSE
  )
  
  ##########################################################
  # 14.2 Simulate one dataset
  ##########################################################
  
  set.seed(structure_seed)
  
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
    stop(
      "sim_logvMEM returned dimension ",
      paste(dim(X), collapse = " x "),
      "; expected ",
      N_total,
      " x ",
      d,
      "."
    )
  }
  
  write.csv(
    X,
    file.path(
      structure_dir,
      paste0(
        "simulated_data_",
        Hlag_structure,
        ".csv"
      )
    ),
    row.names = FALSE
  )
  
  X_train <- X[
    seq_len(N_train),
    ,
    drop = FALSE
  ]
  
  X_test <- X[
    N_total,
    ,
    drop = FALSE
  ]
  
  ##########################################################
  # 14.3 Initial values
  ##########################################################
  
  initial_values <- init.val(
    X = X_train,
    maxlag = p,
    n_beta = 20
  )
  
  init_A_matrix <- initial_values$init_A_matrix
  init_beta <- initial_values$beta_init
  
  if (is.null(init_beta)) {
    
    if (!is.null(initial_values$beta_list)) {
      init_beta <- median(
        initial_values$beta_list,
        na.rm = TRUE
      )
    } else {
      init_beta <- 2
    }
  }
  
  ##########################################################
  # 14.4 Unpenalized corrected-likelihood fit
  ##########################################################
  
  unpenalized_rds <- file.path(
    structure_dir,
    paste0(
      "unpenalized_fit_",
      Hlag_structure,
      ".rds"
    )
  )
  
  if (file.exists(unpenalized_rds)) {
    
    cat(
      "Reading existing unpenalized fit:",
      unpenalized_rds,
      "\n"
    )
    
    fit_unpen <- readRDS(
      unpenalized_rds
    )
    
  } else {
    
    cat(
      "Running unpenalized BCD-BOBYQA fit...\n"
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
    
    saveRDS(
      fit_unpen,
      unpenalized_rds
    )
  }
  
  if (is.null(fit_unpen$A_estimate)) {
    stop(
      "The unpenalized fit does not contain A_estimate."
    )
  }
  
  if (is.null(fit_unpen$beta_estimate)) {
    stop(
      "The unpenalized fit does not contain beta_estimate."
    )
  }
  
  A_matrix_estimated_mle <- fit_unpen$A_estimate
  beta_mle <- fit_unpen$beta_estimate
  
  cat(
    "Estimated beta:",
    beta_mle,
    "\n"
  )
  
  cat(
    "Unpenalized convergence:",
    fit_unpen$converged,
    "\n"
  )
  
  ##########################################################
  # 14.5 Construct lambda path
  ##########################################################
  
  loglik_mle <- log_lik_logvMEM(
    X = X_train,
    A_matrix = A_matrix_estimated_mle,
    beta_par = beta_mle
  )
  
  lambda_max <- abs(
    loglik_mle
  ) / (
    nrow(X_train) *
      p *
      v
  )
  
  if (
    !is.finite(lambda_max) ||
    lambda_max <= 0
  ) {
    stop(
      "Invalid lambda_max: ",
      lambda_max
    )
  }
  
  lambda_grid <- round(
    exp(
      seq(
        log(lambda_max),
        log(
          lambda_max *
            epsilon_lambda
        ),
        length.out = K_lambda
      )
    ),
    digits = 10
  )
  
  write.csv(
    data.frame(
      lambda = lambda_grid
    ),
    file.path(
      structure_dir,
      paste0(
        "lambda_grid_",
        Hlag_structure,
        ".csv"
      )
    ),
    row.names = FALSE
  )
  
  ##########################################################
  # 14.6 Save true support plot
  ##########################################################
  
  A_true_support <- make_support_matrix(
    A_matrix_true,
    threshold = threshold
  )
  
  true_plot_file <- file.path(
    structure_dir,
    paste0(
      "true_sparsity_",
      Hlag_structure,
      ".pdf"
    )
  )
  
  grDevices::pdf(
    true_plot_file,
    width = 14,
    height = 2.7
  )
  
  print(
    SparsityPlot(
      B = A_true_support,
      p = p,
      k = d,
      title = NULL
    )
  )
  
  grDevices::dev.off()
  
  ##########################################################
  # 14.7 Fit each penalty and select lambda by BIC
  ##########################################################
  
  structure_results <- vector(
    "list",
    length(all_penalties)
  )
  
  names(structure_results) <- all_penalties
  
  support_results <- vector(
    "list",
    length(all_penalties)
  )
  
  names(support_results) <- all_penalties
  
  for (penalty in all_penalties) {
    
    cat(
      "\n--------------------------------------------\n"
    )
    
    cat(
      "Structure:",
      Hlag_structure,
      "| Penalty:",
      penalty,
      "\n"
    )
    
    cat(
      "--------------------------------------------\n"
    )
    
    penalty_rds <- file.path(
      structure_dir,
      paste0(
        "penalized_grid_",
        Hlag_structure,
        "_",
        penalty,
        ".rds"
      )
    )
    
    if (file.exists(penalty_rds)) {
      
      cat(
        "Reading saved penalized grid:\n",
        penalty_rds,
        "\n"
      )
      
      sparse_result <- readRDS(
        penalty_rds
      )
      
    } else {
      
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
        progress_file = NULL
      )
      
      saveRDS(
        sparse_result,
        penalty_rds
      )
    }
    
    write.csv(
      sparse_result$summary_df,
      file.path(
        structure_dir,
        paste0(
          "all_lambda_results_",
          Hlag_structure,
          "_",
          penalty,
          ".csv"
        )
      ),
      row.names = FALSE
    )
    
    ########################################################
    # Select minimum-BIC model
    ########################################################
    
    best_result <- extract_best_bic_fit(
      sparse_result = sparse_result,
      penalty = penalty,
      Hlag_structure = Hlag_structure
    )
    
    best_fit <- best_result$fit
    best_summary <- best_result$summary
    
    A_penalized_estimate <- as.matrix(
      best_fit$A_estimate
    )
    
    if (
      !all(
        dim(A_penalized_estimate) ==
        dim(A_matrix_true)
      )
    ) {
      stop(
        "Selected A estimate has dimension ",
        paste(
          dim(A_penalized_estimate),
          collapse = " x "
        ),
        "; true A has dimension ",
        paste(
          dim(A_matrix_true),
          collapse = " x "
        ),
        "."
      )
    }
    
    ########################################################
    # Save coefficient estimate
    ########################################################
    
    write.csv(
      A_penalized_estimate,
      file.path(
        structure_dir,
        paste0(
          "A_estimated_BIC_",
          Hlag_structure,
          "_",
          penalty,
          ".csv"
        )
      ),
      row.names = FALSE
    )
    
    write.csv(
      best_summary,
      file.path(
        structure_dir,
        paste0(
          "best_BIC_summary_",
          Hlag_structure,
          "_",
          penalty,
          ".csv"
        )
      ),
      row.names = FALSE
    )
    
    ########################################################
    # Support recovery
    ########################################################
    
    support_metrics <- calculate_support_metrics(
      A_estimated = A_penalized_estimate,
      A_true = A_matrix_true,
      threshold = threshold
    )
    
    support_metrics$Hlag_structure <- Hlag_structure
    support_metrics$penalty <- penalty
    
    if ("lambda" %in% names(best_summary)) {
      support_metrics$lambda <- best_summary$lambda
    } else {
      support_metrics$lambda <- NA_real_
    }
    
    support_metrics$bic_full_data <-
      best_summary$bic_full_data
    
    support_results[[penalty]] <-
      support_metrics
    
    ########################################################
    # Save estimated support plot
    ########################################################
    
    A_estimated_support <- make_support_matrix(
      A_penalized_estimate,
      threshold = threshold
    )
    
    plot_file <- file.path(
      structure_dir,
      paste0(
        "estimated_sparsity_BIC_",
        Hlag_structure,
        "_",
        penalty,
        ".pdf"
      )
    )
    
    grDevices::pdf(
      plot_file,
      width = 14,
      height = 2.7
    )
    
    print(
      SparsityPlot(
        B = A_estimated_support,
        p = p,
        k = d,
        title = NULL
      )
    )
    
    grDevices::dev.off()
    
    structure_results[[penalty]] <- list(
      sparse_result = sparse_result,
      best_result = best_result,
      A_estimated = A_penalized_estimate,
      support = A_estimated_support,
      support_metrics = support_metrics
    )
  }
  
  ##########################################################
  # 14.8 Save support metrics for structure
  ##########################################################
  
  support_table <- do.call(
    rbind,
    support_results
  )
  
  rownames(support_table) <- NULL
  
  support_table <- support_table[
    ,
    c(
      "Hlag_structure",
      "penalty",
      "lambda",
      "bic_full_data",
      "TP",
      "TN",
      "FP",
      "FN",
      "TPR",
      "FPR",
      "Precision",
      "F1",
      "SupportAccuracy",
      "ExactRecovery"
    )
  ]
  
  write.csv(
    support_table,
    file.path(
      structure_dir,
      paste0(
        "support_recovery_BIC_",
        Hlag_structure,
        ".csv"
      )
    ),
    row.names = FALSE
  )
  
  structure_end_time <- Sys.time()
  
  runtime_seconds <- as.numeric(
    difftime(
      structure_end_time,
      structure_start_time,
      units = "secs"
    )
  )
  
  runtime_table <- data.frame(
    Hlag_structure = Hlag_structure,
    start_time = as.character(
      structure_start_time
    ),
    end_time = as.character(
      structure_end_time
    ),
    runtime_seconds = runtime_seconds,
    runtime_hours = runtime_seconds / 3600,
    beta_true = beta_true,
    beta_estimated_unpenalized = beta_mle,
    unpenalized_converged = fit_unpen$converged,
    stringsAsFactors = FALSE
  )
  
  write.csv(
    runtime_table,
    file.path(
      structure_dir,
      paste0(
        "runtime_",
        Hlag_structure,
        ".csv"
      )
    ),
    row.names = FALSE
  )
  
  list(
    structure = Hlag_structure,
    L = L,
    A_true = A_matrix_true,
    true_support = A_true_support,
    X = X,
    fit_unpen = fit_unpen,
    lambda_grid = lambda_grid,
    penalty_results = structure_results,
    support_table = support_table,
    runtime = runtime_table
  )
}

############################################################
# 15. RUN ALL THREE STRUCTURES
############################################################

study_start_time <- Sys.time()

all_results <- vector(
  "list",
  length(all_structures)
)

names(all_results) <- all_structures

for (structure_id in seq_along(all_structures)) {
  
  Hlag_structure <- all_structures[structure_id]
  
  # The same base seed is retained, but shifted slightly so that
  # each structural design has its own reproducible dataset.
  structure_seed <- simulation_seed + structure_id - 1
  
  all_results[[Hlag_structure]] <-
    run_one_structure(
      Hlag_structure = Hlag_structure,
      structure_seed = structure_seed
    )
  
  saveRDS(
    all_results,
    file.path(
      output_root,
      "all_simulation_results_partial.rds"
    )
  )
}

############################################################
# 16. COMBINE SUPPORT-RECOVERY RESULTS
############################################################

all_support_results <- do.call(
  rbind,
  lapply(
    all_results,
    function(x) x$support_table
  )
)

rownames(all_support_results) <- NULL

write.csv(
  all_support_results,
  file.path(
    output_root,
    "support_recovery_all_structures_BIC.csv"
  ),
  row.names = FALSE
)

############################################################
# 17. SAVE COMPLETE RESULT OBJECT
############################################################

saveRDS(
  all_results,
  file.path(
    output_root,
    "all_simulation_results_complete.rds"
  )
)

study_end_time <- Sys.time()

cat(
  "\n====================================================\n"
)

cat(
  "Simulation study completed.\n"
)

cat(
  "Start time:",
  as.character(study_start_time),
  "\n"
)

cat(
  "End time:",
  as.character(study_end_time),
  "\n"
)

cat(
  "Elapsed time:",
  as.character(
    study_end_time -
      study_start_time
  ),
  "\n"
)

cat(
  "Output directory:",
  output_root,
  "\n"
)

cat(
  "====================================================\n"
)

print(
  all_support_results
)