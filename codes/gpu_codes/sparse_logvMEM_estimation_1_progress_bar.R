############################################################
# Sparse log-vMEM with corrected likelihood + new Method 1
# Uses full corrected likelihood and row-wise penalized updates
############################################################

###################
# Source codes:
###################
project_dir <- "/Users/rohan/Documents/Rohan/Finance paper/Sparse-Logarithmic-Vector-Multiplicative-Error-Model-with-Multivariate-Gamma-Errors-develop_RCodes/"
setwd(project_dir)
source("codes/gpu/mle_logvMEM_1_gpu_test.R")

###################
# Packages:
###################
library(nloptr)
library(ACDm)
library(Rsolnp)

############################################################
# Helper: Frobenius / vector norm
############################################################
norm_vec <- function(x) sqrt(sum(x^2))

############################################################
# Helper: compute omega from A, beta, and data
# Matches the full corrected likelihood setup
############################################################
compute_omega_from_A <- function(X, A_matrix, beta_par, p) {
  d <- ncol(X)
  
  all_param <- unlist(lapply(1:ncol(A_matrix), function(col.index) A_matrix[, col.index]))
  A_matrices <- all_param[1:(p * d^2)]
  
  start_id <- seq(1, p * d^2, by = d^2)
  end_id   <- seq(d^2, p * d^2, by = d^2)
  
  A_matrix_list <- lapply(1:p, function(i) {
    matrix(A_matrices[start_id[i]:end_id[i]], nrow = d, ncol = d)
  })
  
  mu_vec <- colMeans(X)
  variance_vec <- diag(cov(X))
  M <- rep(digamma(beta_par), d) - log(beta_par)
  
  omega <- crossprod(
    t(diag(1, d) - Reduce(`+`, A_matrix_list)),
    matrix(log(mu_vec) - variance_vec / (2 * mu_vec^2))
  ) - M
  
  as.numeric(omega)
}

############################################################
# Full fitted values for log-vMEM under corrected likelihood
############################################################
fitted_values_logvMEM_full <- function(X, A_matrix, beta_par, p) {
  N <- nrow(X)
  d <- ncol(X)
  
  omega <- compute_omega_from_A(X = X, A_matrix = A_matrix, beta_par = beta_par, p = p)
  
  all_param <- unlist(lapply(1:ncol(A_matrix), function(col.index) A_matrix[, col.index]))
  A_matrices <- all_param[1:(p * d^2)]
  
  start_id <- seq(1, p * d^2, by = d^2)
  end_id   <- seq(d^2, p * d^2, by = d^2)
  
  A_matrix_list <- lapply(1:p, function(i) {
    matrix(A_matrices[start_id[i]:end_id[i]], nrow = d, ncol = d)
  })
  
  log_mu_t <- matrix(NA_real_, nrow = N, ncol = d)
  log_mu_t[1:p, ] <- rep(0.1, p)
  
  log_x_t <- log(X)
  
  log_mu_t[(p + 1):N, ] <- do.call(rbind, lapply((p + 1):N, function(tt) {
    t(omega + list.multiply(
      list1 = A_matrix_list,
      list2 = generate_lag_matrix(x_t = log_x_t, t = tt, lag_order = p)
    ))
  }))
  
  exp(log_mu_t)
}

############################################################
# Updated h-step forecast function
############################################################
forecast_logvMEM_full_data_new <- function(X_train, X_test, h_step,
                                           A_matrix_estimated,
                                           beta_estimated,
                                           maxlag) {
  p <- maxlag
  d <- ncol(X_train)
  n_t <- nrow(X_train)
  
  omega_estimated <- compute_omega_from_A(
    X = X_train,
    A_matrix = A_matrix_estimated,
    beta_par = beta_estimated,
    p = p
  )
  
  model_data <- as.data.frame(X_train)
  
  start_id <- seq(1, ncol(A_matrix_estimated), by = d)
  end_id   <- seq(d, ncol(A_matrix_estimated), by = d)
  
  A_matrices_list <- lapply(seq_along(start_id), function(j) {
    as.matrix(A_matrix_estimated[, start_id[j]:end_id[j], drop = FALSE])
  })
  
  for (forecast_id in 1:h_step) {
    indices <- (n_t + forecast_id - 1):(n_t + forecast_id - p)
    
    model_data[n_t + forecast_id, ] <- exp(
      omega_estimated +
        Reduce(`+`, lapply(1:p, function(id) {
          A_matrices_list[[id]] %*% matrix(as.numeric(log(model_data[indices[id], ])), ncol = 1)
        }))
    )
  }
  
  forecast_data <- as.matrix(model_data[(n_t + 1):(n_t + h_step), , drop = FALSE])
  actual_data   <- matrix(X_test, nrow = h_step)
  
  msfe <- mean(unlist(lapply(1:ncol(actual_data), function(id) {
    mean((actual_data[, id] - forecast_data[, id])^2)
  })))
  
  mad <- mean(unlist(lapply(1:ncol(actual_data), function(id) {
    mean(abs(actual_data[, id] - forecast_data[, id]))
  })))
  
  mape <- mean(unlist(lapply(1:ncol(actual_data), function(id) {
    mean(abs((actual_data[, id] - forecast_data[, id]) / actual_data[, id]))
  })))
  
  df <- cbind.data.frame(
    actual   = c(t(actual_data)),
    forecast = c(t(forecast_data))
  )
  df$MSFE_i <- (df$actual - df$forecast)^2
  df$MAD_i  <- abs(df$actual - df$forecast)
  df$MAPE_i <- abs((df$actual - df$forecast) / df$actual)
  
  list(
    model_metrics = c(MSFE = msfe, MAD = mad, MAPE = mape),
    forecast_df = df
  )
}

############################################################
# Penalty functions
############################################################
scad <- function(theta, gamma_par, lambda_par) {
  gamma_lambda <- gamma_par * lambda_par
  if (theta <= lambda_par) {
    val <- lambda_par * theta
  } else if (theta <= gamma_lambda) {
    val <- (gamma_par * lambda_par * theta - 0.5 * (theta^2 + lambda_par^2)) / (gamma_par - 1)
  } else {
    val <- lambda_par^2 * (gamma_par + 1) / 2
  }
  val
}

mcp <- function(theta, gamma_par, lambda_par) {
  gamma_lambda <- gamma_par * lambda_par
  if (abs(theta) <= gamma_lambda) {
    val <- lambda_par * abs(theta) - theta^2 / (2 * gamma_par)
  } else {
    val <- gamma_par * lambda_par^2 / 2
  }
  val
}

############################################################
# Row-wise penalty g_i
# Preserves the old penalty structures, but now used with the
# full corrected likelihood during row updates
############################################################
g_i_new <- function(i.index, A_matrix_i, lambda_penalty,
                    penalty, Hlag_structure, mle_i_vec, d, p) {
  
  if (is.null(mle_i_vec)) mle_i_vec <- rep(1, length(A_matrix_i))
  
  ####################
  # Componentwise
  ####################
  if (penalty == "group-lasso" && Hlag_structure == "componentwise") {
    A_mat <- A_matrix_i
    group_list <- lapply(seq(1, d * p, by = d), function(x) x:(d * p))
    ret_value <- sum(unlist(lapply(group_list, function(x) norm_vec(A_mat[x])))) * lambda_penalty
    return(ret_value)
  }
  
  if (penalty == "adaptive-group-lasso" && Hlag_structure == "componentwise") {
    gamma_par <- 1
    A_mat <- A_matrix_i
    group_list <- lapply(seq(1, d * p, by = d), function(x) x:(d * p))
    A_mat_norm_groupwise  <- unlist(lapply(group_list, function(x) norm_vec(A_mat[x])))
    mle_i_norm_groupwise  <- unlist(lapply(group_list, function(x) norm_vec(mle_i_vec[x])))
    mle_i_norm_groupwise[mle_i_norm_groupwise < 0.005] <- 0.005
    ret_value <- sum(mle_i_norm_groupwise^(-gamma_par) * A_mat_norm_groupwise) * lambda_penalty
    return(ret_value)
  }
  
  if (penalty == "group-scad" && Hlag_structure == "componentwise") {
    A_mat <- A_matrix_i
    group_list <- lapply(seq(1, d * p, by = d), function(x) x:(d * p))
    ret_value <- sum(unlist(lapply(group_list, function(x) {
      scad(norm_vec(A_mat[x]), gamma_par = 3.7, lambda_par = lambda_penalty)
    })))
    return(ret_value)
  }
  
  if (penalty == "group-mcp" && Hlag_structure == "componentwise") {
    A_mat <- A_matrix_i
    group_list <- lapply(seq(1, d * p, by = d), function(x) x:(d * p))
    ret_value <- sum(unlist(lapply(group_list, function(x) {
      mcp(norm_vec(A_mat[x]), gamma_par = 3, lambda_par = lambda_penalty)
    })))
    return(ret_value)
  }
  
  ####################
  # Elementwise
  ####################
  if (penalty == "group-lasso" && Hlag_structure == "elementwise") {
    A_mat_list <- lapply(seq(1, p * d, by = d), function(x) as.numeric(A_matrix_i[x:(x + d - 1)]))
    new_A_mat <- do.call(rbind, A_mat_list)
    grouped_cal <- rep(NA, ncol(new_A_mat))
    for (j in 1:ncol(new_A_mat)) {
      grouped_cal[j] <- sum(unlist(lapply(seq(1, p), function(x) norm_vec(new_A_mat[, j][x:p]))))
    }
    return(sum(grouped_cal) * lambda_penalty)
  }
  
  if (penalty == "adaptive-group-lasso" && Hlag_structure == "elementwise") {
    gamma_par <- 1
    A_mat_list  <- lapply(seq(1, p * d, by = d), function(x) as.numeric(A_matrix_i[x:(x + d - 1)]))
    mle_i_list  <- lapply(seq(1, p * d, by = d), function(x) as.numeric(mle_i_vec[x:(x + d - 1)]))
    new_A_mat   <- do.call(rbind, A_mat_list)
    new_mle_mat <- do.call(rbind, mle_i_list)
    new_mle_mat[new_mle_mat < 0.005] <- 0.005
    grouped_cal <- rep(NA, ncol(new_A_mat))
    for (j in 1:ncol(new_A_mat)) {
      grouped_cal[j] <- sum(unlist(lapply(seq(1, p), function(x) {
        norm_vec(new_A_mat[, j][x:p]) * norm_vec(new_mle_mat[, j][x:p])^(-gamma_par)
      })))
    }
    return(sum(grouped_cal) * lambda_penalty)
  }
  
  if (penalty == "group-scad" && Hlag_structure == "elementwise") {
    A_mat_list <- lapply(seq(1, p * d, by = d), function(x) as.numeric(A_matrix_i[x:(x + d - 1)]))
    new_A_mat <- do.call(rbind, A_mat_list)
    grouped_cal <- rep(NA, ncol(new_A_mat))
    for (j in 1:ncol(new_A_mat)) {
      grouped_cal[j] <- sum(unlist(lapply(seq(1, p), function(x) {
        scad(norm_vec(new_A_mat[, j][x:p]), gamma_par = 3.7, lambda_par = lambda_penalty)
      })))
    }
    return(sum(grouped_cal))
  }
  
  if (penalty == "group-mcp" && Hlag_structure == "elementwise") {
    A_mat_list <- lapply(seq(1, p * d, by = d), function(x) as.numeric(A_matrix_i[x:(x + d - 1)]))
    new_A_mat <- do.call(rbind, A_mat_list)
    grouped_cal <- rep(NA, ncol(new_A_mat))
    for (j in 1:ncol(new_A_mat)) {
      grouped_cal[j] <- sum(unlist(lapply(seq(1, p), function(x) {
        mcp(norm_vec(new_A_mat[, j][x:p]), gamma_par = 3, lambda_par = lambda_penalty)
      })))
    }
    return(sum(grouped_cal))
  }
  
  ####################
  # Own-other
  ####################
  if (penalty == "group-lasso" && Hlag_structure == "own-other") {
    A_mat <- A_matrix_i
    group_list <- lapply(seq(1, d * p, by = d), function(x) x:(d * p))
    componentwise_sum <- sum(unlist(lapply(group_list, function(x) norm_vec(A_mat[x]))))
    A_mat_list <- lapply(seq(1, p * d, by = d), function(x) A_matrix_i[x:(x + d - 1)])
    new_l <- list()
    for (l in 1:(p - 1)) {
      new_l[[l]] <- norm_vec(c(as.numeric(A_mat_list[[l]][-i.index]),
                               as.numeric(sapply((l + 1):p, function(x) A_mat_list[[x]]))))
    }
    own_other_sum <- sum(unlist(new_l))
    return((componentwise_sum + own_other_sum) * lambda_penalty)
  }
  
  if (penalty == "adaptive-group-lasso" && Hlag_structure == "own-other") {
    gamma_par <- 1
    A_mat <- A_matrix_i
    group_list <- lapply(seq(1, d * p, by = d), function(x) x:(d * p))
    A_mat_norm_groupwise <- unlist(lapply(group_list, function(x) norm_vec(A_mat[x])))
    mle_i_norm_groupwise <- unlist(lapply(group_list, function(x) norm_vec(mle_i_vec[x])))
    mle_i_norm_groupwise[mle_i_norm_groupwise < 0.005] <- 0.005
    componentwise_sum <- sum(mle_i_norm_groupwise^(-gamma_par) * A_mat_norm_groupwise)
    
    A_mat_list <- lapply(seq(1, p * d, by = d), function(x) A_matrix_i[x:(x + d - 1)])
    mle_tmp <- mle_i_vec
    mle_tmp[mle_tmp < 0.005] <- 0.005
    mle_i_mat_list <- lapply(seq(1, p * d, by = d), function(x) mle_tmp[x:(x + d - 1)])
    
    new_l <- list()
    for (l in 1:(p - 1)) {
      num <- norm_vec(c(as.numeric(A_mat_list[[l]][-i.index]),
                        as.numeric(sapply((l + 1):p, function(x) A_mat_list[[x]]))))
      den <- norm_vec(c(as.numeric(mle_i_mat_list[[l]][-i.index]),
                        as.numeric(sapply((l + 1):p, function(x) mle_i_mat_list[[x]]))))
      if (den < 0.005) den <- 0.005
      new_l[[l]] <- den^(-gamma_par) * num
    }
    own_other_sum <- sum(unlist(new_l))
    return((componentwise_sum + own_other_sum) * lambda_penalty)
  }
  
  if (penalty == "group-scad" && Hlag_structure == "own-other") {
    
    A_mat <- A_matrix_i
    
    group_list <- lapply(
      seq(1, d * p, by = d),
      function(x) x:(d * p)
    )
    
    componentwise_sum <- sum(
      unlist(
        lapply(group_list, function(x) {
          scad(
            norm_vec(A_mat[x]),
            gamma_par = 3.7,
            lambda_par = lambda_penalty
          )
        })
      )
    )
    
    A_mat_list <- lapply(
      seq(1, p * d, by = d),
      function(x) A_matrix_i[x:(x + d - 1)]
    )
    
    new_l <- list()
    
    for (l in 1:(p - 1)) {
      
      new_l[[l]] <- scad(
        norm_vec(
          c(
            as.numeric(A_mat_list[[l]][-i.index]),
            as.numeric(
              sapply(
                (l + 1):p,
                function(x) A_mat_list[[x]]
              )
            )
          )
        ),
        gamma_par = 3.7,
        lambda_par = lambda_penalty
      )
    }
    
    own_other_sum <- sum(unlist(new_l))
    
    return(componentwise_sum + own_other_sum)
  }
  
  if (penalty == "group-mcp" && Hlag_structure == "own-other") {
    A_mat <- A_matrix_i
    group_list <- lapply(seq(1, d * p, by = d), function(x) x:(d * p))
    componentwise_sum <- sum(unlist(lapply(group_list, function(x) {
      mcp(norm_vec(A_mat[x]), gamma_par = 3, lambda_par = lambda_penalty)
    })))
    A_mat_list <- lapply(seq(1, p * d, by = d), function(x) A_matrix_i[x:(x + d - 1)])
    new_l <- list()
    for (l in 1:(p - 1)) {
      new_l[[l]] <- mcp(
        norm_vec(c(as.numeric(A_mat_list[[l]][-i.index]),
                   as.numeric(sapply((l + 1):p, function(x) A_mat_list[[x]])))),
        gamma_par = 3, lambda_par = lambda_penalty
      )
    }
    own_other_sum <- sum(unlist(new_l))
    return(componentwise_sum + own_other_sum)
  }
  
  0
}

############################################################
# Penalized row update using full corrected likelihood
############################################################
############################################################
# Penalized row update using resident CUDA likelihood
#
# BOBYQA remains CPU-side.
# The likelihood is evaluated on CUDA using the persistent
# context created once for the full fixed-beta A update.
############################################################

update_A_block_penalized <- function(
    A_current,
    i.index,
    cuda_context,
    lambda_penalty,
    penalty,
    Hlag_structure,
    mle_A_matrix = NULL,
    lower = -0.5,
    upper = 1,
    max_iter_bobyqa = 1000,
    eps = 1e-4
) {
  
  A_current <- as.matrix(A_current)
  
  d <- cuda_context$d
  p <- cuda_context$p
  N <- cuda_context$N
  
  stopifnot(nrow(A_current) == d)
  stopifnot(ncol(A_current) == p * d)
  
  lb <- rep(lower, p * d)
  ub <- rep(upper, p * d)
  
  mle_i_vec <- if (is.null(mle_A_matrix)) {
    NULL
  } else {
    as.numeric(mle_A_matrix[i.index, ])
  }
  
  # ==========================================================
  # PENALIZED OBJECTIVE
  #
  # -(1/N) log L(A,beta) + penalty(A_i)
  # ==========================================================
  
  obj_fun <- function(a_i_vec) {
    
    A_trial <- A_current
    A_trial[i.index, ] <- a_i_vec
    
    loglik <- log_lik_logvMEM_cuda_context(
      A_matrix = A_trial,
      context = cuda_context
    )
    
    if (!is.finite(loglik)) {
      return(1e12)
    }
    
    neg_ll <- (-1 / N) * loglik
    
    pen_val <- g_i_new(
      i.index = i.index,
      A_matrix_i = a_i_vec,
      lambda_penalty = lambda_penalty,
      penalty = penalty,
      Hlag_structure = Hlag_structure,
      mle_i_vec = mle_i_vec,
      d = d,
      p = p
    )
    
    val <- neg_ll + pen_val
    
    if (!is.finite(val)) {
      val <- 1e12
    }
    
    val
  }
  
  # ==========================================================
  # CPU BOBYQA OPTIMIZATION
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
  # FALLBACK
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
  # RETURN
  # ==========================================================
  
  list(
    A_i_estimate = fit$par,
    convergence = fit$convergence,
    iter = fit$iter,
    objective = fit$value
  )
}

############################################################
# Penalized blockwise update of A
############################################################
############################################################
# Penalized blockwise A update using resident CUDA context
#
# One CUDA context is created for the entire fixed-beta
# A optimization and reused across:
#
#   - all BOBYQA evaluations
#   - all rows of A
#   - all Gauss-Seidel sweeps
############################################################

update_A_blockwise_penalized <- function(
    X,
    A_init,
    beta_par,
    lambda_penalty,
    penalty,
    Hlag_structure,
    mle_A_matrix = NULL,
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
  # CREATE ONE PERSISTENT CUDA CONTEXT
  #
  # beta is fixed throughout this entire A update.
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
      "  Penalized CUDA context created:",
      "d =", d,
      "| p =", p,
      "| beta =", beta_par,
      "| lambda =", lambda_penalty,
      "| penalty =", penalty,
      "| Hlag =", Hlag_structure,
      "| Q =", n_quad,
      "\n"
    )
  }
  
  block_log <- vector(
    "list",
    max_block_sweeps
  )
  
  # ==========================================================
  # GAUSS-SEIDEL SWEEPS
  # ==========================================================
  
  for (sweep in seq_len(max_block_sweeps)) {
    
    if (trace) {
      cat(
        "  Penalized A-block sweep:",
        sweep,
        "\n"
      )
    }
    
    row_log <- vector(
      "list",
      d
    )
    
    for (i in seq_len(d)) {
      
      upd <- update_A_block_penalized(
        A_current = A_current,
        i.index = i,
        cuda_context = cuda_context,
        lambda_penalty = lambda_penalty,
        penalty = penalty,
        Hlag_structure = Hlag_structure,
        mle_A_matrix = mle_A_matrix,
        lower = lower,
        upper = upper,
        max_iter_bobyqa = max_iter_bobyqa,
        eps = eps
      )
      
      # Gauss-Seidel:
      # immediately use the updated row
      A_current[i, ] <- upd$A_i_estimate
      
      row_log[[i]] <- upd
    }
    
    # ========================================================
    # UNPENALIZED LOG-LIKELIHOOD AFTER SWEEP
    #
    # This is intentionally the likelihood only, not the
    # penalized objective.
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
        "    loglik after penalized sweep =",
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
  
  list(
    A_estimate = A_current,
    loglik = final_loglik,
    block_log = block_log
  )
}

############################################################
# Penalized Method 1:
# alternating penalized A-updates + beta-update
# Note: penalty is on A only, so beta-update is unchanged
############################################################
############################################################
# Penalized Method 1 using CUDA
#
# Alternating optimization:
#
#   1. penalized A update | beta fixed
#   2. beta update        | A fixed
#
# Penalty applies only to A.
############################################################

estimate_penalized_method1_joint <- function(
    X,
    init_A_matrix,
    init_beta,
    lambda_penalty,
    penalty,
    Hlag_structure,
    mle_A_matrix = NULL,
    max_outer_iter = 10,
    max_block_sweeps = 1,
    lower_A = -0.5,
    upper_A = 1,
    beta_lower = 0.2,
    beta_upper = 20,
    max_iter_bobyqa = 1000,
    eps_A = 1e-4,
    tol_beta = 1e-3,
    tol_rel_loglik = 1e-5,
    rel_loglik_patience = 2L,
    n_quad = 64L,
    device = "cuda",
    trace = TRUE
) {
  
  X <- as.matrix(X)
  A_current <- as.matrix(init_A_matrix)
  beta_current <- init_beta
  
  d <- ncol(X)
  p <- ncol(A_current) / d
  
  stopifnot(nrow(A_current) == d)
  stopifnot(ncol(A_current) == p * d)
  
  history <- vector(
    "list",
    max_outer_iter
  )
  
  converged <- FALSE
  stable_loglik_count <- 0L
  
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
  # ==========================================================
  # OUTER ALTERNATING LOOP
  # ==========================================================
  
  for (iter in seq_len(max_outer_iter)) {
    
    if (trace) {
      cat(
        "\nPenalized outer iteration:",
        iter,
        "\n"
      )
    }
    
    A_old <- A_current
    beta_old <- beta_current
    
    # ========================================================
    # STEP 1: PENALIZED A UPDATE | beta fixed
    # ========================================================
    
    A_upd <- update_A_blockwise_penalized(
      X = X,
      A_init = A_current,
      beta_par = beta_current,
      lambda_penalty = lambda_penalty,
      penalty = penalty,
      Hlag_structure = Hlag_structure,
      mle_A_matrix = mle_A_matrix,
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
    
    ll_after_A <- A_upd$loglik
    
    # ========================================================
    # STEP 2: beta UPDATE | A fixed
    #
    # Penalty contains no beta, so this is exactly the same
    # beta optimization used by the unpenalized estimator.
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
    
    # beta changed -> create beta-specific context
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
    ll_old <- ll_current
    
    rel_loglik_change <- abs(
      ll_new - ll_old
    ) / max(
      1,
      abs(ll_old)
    )
    # ========================================================
    # CONVERGENCE
    # Preserve the same Frobenius norm used by the CPU sparse
    # estimator so behavior remains comparable.
    # ========================================================
    
    delta_A <- sqrt(
      sum(
        (A_current - A_old)^2
      )
    )
    
    delta_beta <- abs(
      beta_current - beta_old
    )
    
    history[[iter]] <- list(
      iter = iter,
      A = A_current,
      beta = beta_current,
      loglik_after_A = ll_after_A,
      loglik = ll_new,
      rel_loglik_change = rel_loglik_change,
      delta_A = delta_A,
      delta_beta = delta_beta,
      A_update_log = A_upd$block_log,
      beta_objective = beta_upd$objective
    )
    
    if (trace) {
      cat(
        "  Updated beta =",
        beta_current,
        "\n"
      )
      
      cat(
        "  Updated loglik =",
        ll_new,
        "\n"
      )
      
      cat(
        "  relative loglik change =",
        rel_loglik_change,
        "\n"
      )
      
      cat(
        "  delta_A =",
        delta_A,
        "\n"
      )
      
      cat(
        "  delta_beta =",
        delta_beta,
        "\n"
      )
    }
    
    if (rel_loglik_change < tol_rel_loglik) {
      stable_loglik_count <- stable_loglik_count + 1L
    } else {
      stable_loglik_count <- 0L
    }
    
    ll_current <- ll_new
    
    if (stable_loglik_count >= rel_loglik_patience) {
      
      converged <- TRUE
      
      if (trace) {
        cat(
          "Penalized estimator converged after",
          stable_loglik_count,
          "consecutive stable likelihood iterations.\n"
        )
      }
      
      history <- history[
        seq_len(iter)
      ]
      
      break
    }
  }
  
  n_outer_iter <- iter
  
  # ==========================================================
  # RETURN
  # ==========================================================
  
  list(
    A_estimate = A_current,
    beta_estimate = beta_current,
    loglik = ll_new,
    n_outer_iter = n_outer_iter,
    converged = converged,
    history = history
  )
}

############################################################
# Evaluate one penalized fit
############################################################
############################################################
# Evaluate one penalized GPU fit
############################################################

evaluate_penalized_fit <- function(
    X_train,
    X_test,
    h_step,
    fit_pen,
    penalty,
    Hlag_structure,
    lambda_penalty,
    threshold = 1e-4,
    maxlag,
    n_quad = 64L,
    device = "cuda"
) {
  
  A_hat <- fit_pen$A_estimate
  beta_hat <- fit_pen$beta_estimate
  
  N <- nrow(X_train)
  
  # ==========================================================
  # GPU FULL LOG-LIKELIHOOD
  # ==========================================================
  
  eval_context <- create_logvMEM_cuda_context(
    X = X_train,
    p = maxlag,
    beta_par = beta_hat,
    n_quad = n_quad,
    device = device
  )
  
  loglik_full <- log_lik_logvMEM_cuda_context(
    A_matrix = A_hat,
    context = eval_context
  )
  
  # ==========================================================
  # INFORMATION CRITERIA
  # ==========================================================
  
  df_full <- sum(
    abs(A_hat) > threshold
  )
  
  aic_full <- (
    2 * df_full -
      2 * loglik_full
  )
  
  bic_full <- (
    df_full * log(N) -
      2 * loglik_full
  )
  
  # ==========================================================
  # IN-SAMPLE FIT
  # ==========================================================
  
  fitted_vals <- fitted_values_logvMEM_full(
    X = X_train,
    A_matrix = A_hat,
    beta_par = beta_hat,
    p = maxlag
  )
  
  mad_insample <- mean(
    abs(fitted_vals - X_train),
    na.rm = TRUE
  )
  
  # ==========================================================
  # OUT-OF-SAMPLE FORECAST
  # ==========================================================
  
  fc_obj <- forecast_logvMEM_full_data_new(
    X_train = X_train,
    X_test = X_test,
    h_step = h_step,
    A_matrix_estimated = A_hat,
    beta_estimated = beta_hat,
    maxlag = maxlag
  )
  
  # ==========================================================
  # RETURN
  # ==========================================================
  
  list(
    summary_df = data.frame(
      penalty = penalty,
      HLag_structure = Hlag_structure,
      lambda = lambda_penalty,
      beta = beta_hat,
      loglik_full_data = loglik_full,
      df_full_data = df_full,
      aic_full_data = aic_full,
      bic_full_data = bic_full,
      MAD_full_data_insample = mad_insample,
      count_non_zero_coef_full_data = df_full,
      MSFE = as.numeric(
        fc_obj$model_metrics["MSFE"]
      ),
      MAD = as.numeric(
        fc_obj$model_metrics["MAD"]
      ),
      MAPE = as.numeric(
        fc_obj$model_metrics["MAPE"]
      ),
      converged = fit_pen$converged,
      n_outer_iter = fit_pen$n_outer_iter
    ),
    fit = fit_pen,
    forecast = fc_obj
  )
}

############################################################
# Grid search over penalties / lambda
############################################################
run_sparse_grid_new <- function(X_train, X_test, h_step,
                                lambda_grid,
                                penalty_vec,
                                Hlag_vec,
                                maxlag,
                                n_beta = 21,
                                threshold = 1e-4,
                                max_outer_iter = 10,
                                max_block_sweeps = 1,
                                max_iter_bobyqa = 1000,
                                init_A_matrix = NULL,
                                init_beta = NULL,
                                mle_A_matrix = NULL,
                                fit_unpen = NULL,
                                start_from_mle = TRUE,
                                eps_A = 1e-4,
                                tol_beta = 1e-3,
                                tol_rel_loglik = 1e-5,
                                rel_loglik_patience = 2L,
                                n_quad = 64L,
                                device = "cuda",
                                trace = TRUE,
                                progress_file = NULL) {
  
  p <- maxlag
  d <- ncol(X_train)
  
  if (is.null(init_A_matrix) || is.null(init_beta)) {
    if (trace) cat("\nGenerating initial values using init.val()...\n")
    
    init_obj <- init.val(X = X_train, maxlag = p, n_beta = n_beta)
    
    if (is.null(init_A_matrix)) init_A_matrix <- init_obj$init_A_matrix
    if (is.null(init_beta))     init_beta     <- init_obj$beta_init
  }
  
  if (is.null(mle_A_matrix)) {
    
    if (!is.null(fit_unpen)) {
      if (trace) cat("\nUsing supplied unpenalized fit.\n")
      mle_A_matrix <- fit_unpen$A_estimate
      
    } else {
      if (trace) cat("\nRunning unpenalized Method 1 fit first...\n")
      
      fit_unpen <- estimate_method1_joint(
        X = X_train,
        p = p,
        A_init = init_A_matrix,
        beta_init = init_beta,
        max_outer_iter = max_outer_iter,
        max_block_sweeps = max_block_sweeps,
        max_iter_bobyqa = max_iter_bobyqa,
        eps_A = eps_A,
        tol_beta = tol_beta,
        tol_rel_loglik = tol_rel_loglik,
        rel_loglik_patience = rel_loglik_patience,
        n_quad = n_quad,
        device = device,
        trace = trace
      )
      
      mle_A_matrix <- fit_unpen$A_estimate
    }
    
  } else {
    if (trace) cat("\nUsing supplied mle_A_matrix. Skipping unpenalized MLE.\n")
  }
  
  if (start_from_mle) {
    A_start_pen <- mle_A_matrix
  } else {
    A_start_pen <- init_A_matrix
  }
  
  out_list <- list()
  idx <- 1
  total_jobs <- length(penalty_vec) * length(Hlag_vec) * length(lambda_grid)
  
  for (penalty in penalty_vec) {
    for (Hlag_structure in Hlag_vec) {
      for (lambda_penalty in lambda_grid) {
        
        if (!is.null(progress_file)) {
          write_progress_bar(
            progress_file = progress_file,
            i = idx,
            total = total_jobs,
            extra = paste(
              "START lambda =", lambda_penalty,
              "| penalty =", penalty,
              "| Hlag =", Hlag_structure
            )
          )
        }
        
        if (trace) {
          cat("\n====================================\n")
          cat("Job", idx, "of", total_jobs, "\n")
          cat("penalty =", penalty,
              "| Hlag =", Hlag_structure,
              "| lambda =", lambda_penalty, "\n")
          cat("====================================\n")
        }
        
        t1 <- Sys.time()
        
        fit_pen <- estimate_penalized_method1_joint(
          X = X_train,
          init_A_matrix = A_start_pen,
          init_beta = init_beta,
          lambda_penalty = lambda_penalty,
          penalty = penalty,
          Hlag_structure = Hlag_structure,
          mle_A_matrix = mle_A_matrix,
          max_outer_iter = max_outer_iter,
          max_block_sweeps = max_block_sweeps,
          max_iter_bobyqa = max_iter_bobyqa,
          eps_A = eps_A,
          tol_beta = tol_beta,
          tol_rel_loglik = tol_rel_loglik,
          rel_loglik_patience = rel_loglik_patience,
          n_quad = n_quad,
          device = device,
          trace = trace
        )
        
        t2 <- Sys.time()
        
        eval_obj <- evaluate_penalized_fit(
          X_train = X_train,
          X_test = X_test,
          h_step = h_step,
          fit_pen = fit_pen,
          penalty = penalty,
          Hlag_structure = Hlag_structure,
          lambda_penalty = lambda_penalty,
          threshold = threshold,
          maxlag = maxlag,
          n_quad = n_quad,
          device = device
        )
        
        eval_obj$summary_df$runtime_sec <- as.numeric(difftime(t2, t1, units = "secs"))
        
        out_list[[idx]] <- eval_obj
        
        if (!is.null(progress_file)) {
          write_progress_bar(
            progress_file = progress_file,
            i = idx,
            total = total_jobs,
            extra = paste(
              "DONE lambda =", lambda_penalty,
              "| runtime_sec =", round(eval_obj$summary_df$runtime_sec, 2)
            )
          )
        }
        
        if (trace) {
          cat("Finished job", idx, "in",
              round(eval_obj$summary_df$runtime_sec, 2), "seconds.\n")
        }
        
        idx <- idx + 1
      }
    }
  }
  
  summary_df <- do.call(rbind, lapply(out_list, function(x) x$summary_df))
  
  list(
    summary_df = summary_df,
    unpenalized_fit = fit_unpen,
    mle_A_matrix = mle_A_matrix,
    all_fits = out_list
  )
}
######



