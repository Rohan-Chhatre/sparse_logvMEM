############################################################
# CREATE TABLE 5 FROM COMPLETED MODEL OUTPUTS
############################################################

project_dir <- paste0(
  "/Users/rohan/Documents/Rohan/Finance paper/",
  "Sparse-Logarithmic-Vector-Multiplicative-Error-Model-",
  "with-Multivariate-Gamma-Errors-develop_RCodes/"
)

setwd(project_dir)

output_dir <- file.path(
  project_dir,
  "real_data_analysis",
  "output_corrected_gamma",
  "run_3.0"
)

############################################################
# 1. Read all penalty × HLag × lambda results
############################################################

all_results_file <- file.path(
  output_dir,
  "output_table_ALL_penalty_HLag_lambda.csv"
)

if (!file.exists(all_results_file)) {
  stop(
    "Cannot find output_table_ALL_penalty_HLag_lambda.csv"
  )
}

results_all <- read.csv(
  all_results_file,
  stringsAsFactors = FALSE
)

cat("Rows read:", nrow(results_all), "\n")

############################################################
# 2. Check required columns
############################################################

required_columns <- c(
  "penalty",
  "HLag_structure",
  "lambda",
  "bic_full_data",
  "MSFE",
  "MAD",
  "MAPE"
)

missing_columns <- setdiff(
  required_columns,
  names(results_all)
)

if (length(missing_columns) > 0) {
  stop(
    paste(
      "Missing required columns:",
      paste(missing_columns, collapse = ", ")
    )
  )
}

############################################################
# 3. Basic checks
############################################################

cat(
  "Number of penalty-HLag combinations:",
  nrow(
    unique(
      results_all[, c("penalty", "HLag_structure")]
    )
  ),
  "\n"
)

cat(
  "Number of lambda-level results:",
  nrow(results_all),
  "\n"
)

if ("converged" %in% names(results_all)) {
  cat("\nConvergence summary:\n")
  print(
    table(
      results_all$converged,
      useNA = "ifany"
    )
  )
}

if ("n_outer_iter" %in% names(results_all)) {
  cat("\nOuter iteration summary:\n")
  print(
    table(
      results_all$n_outer_iter,
      useNA = "ifany"
    )
  )
}

############################################################
# 4. Remove rows with unusable BIC or forecast measures
############################################################

valid_results <- results_all[
  is.finite(results_all$bic_full_data) &
    is.finite(results_all$MSFE) &
    is.finite(results_all$MAD) &
    is.finite(results_all$MAPE),
  ,
  drop = FALSE
]

if (nrow(valid_results) == 0) {
  stop("No valid model results remain after filtering.")
}

############################################################
# 5. Select lambda using minimum BIC within each combination
############################################################

combination_key <- interaction(
  valid_results$HLag_structure,
  valid_results$penalty,
  drop = TRUE
)

split_results <- split(
  valid_results,
  combination_key
)

bic_selected <- do.call(
  rbind,
  lapply(
    split_results,
    function(x) {
      
      # In case of tied BIC, take the first occurrence
      x[
        which.min(x$bic_full_data),
        ,
        drop = FALSE
      ]
    }
  )
)

rownames(bic_selected) <- NULL

cat(
  "\nBIC-selected penalized models:",
  nrow(bic_selected),
  "\n"
)

############################################################
# 6. Create model labels matching Table 5
############################################################

format_hlag <- function(x) {
  
  output <- x
  
  output[x == "componentwise"] <- "Componentwise"
  output[x == "elementwise"]   <- "Elementwise"
  output[x == "own-other"]     <- "Own-Other"
  
  output
}

format_penalty <- function(x) {
  
  output <- x
  
  output[x == "adaptive-group-lasso"] <-
    "Adaptive Group Lasso"
  
  output[x == "group-lasso"] <-
    "Group Lasso"
  
  output[x == "group-mcp"] <-
    "Group MCP"
  
  output[x == "group-scad"] <-
    "Group SCAD"
  
  output
}

bic_selected$Model <- paste0(
  format_hlag(bic_selected$HLag_structure),
  "-",
  format_penalty(bic_selected$penalty)
)

############################################################
# 7. Read no-penalty benchmark
############################################################

no_penalty_file <- file.path(
  output_dir,
  "final_estimate_MSFE-NoPenalty.csv"
)

if (!file.exists(no_penalty_file)) {
  stop("Cannot find final_estimate_MSFE-NoPenalty.csv")
}

no_penalty_raw <- read.csv(
  no_penalty_file,
  stringsAsFactors = FALSE
)

required_no_penalty_columns <- c(
  "MSFE_forecast",
  "MAD_forecast",
  "MAPE_forecast"
)

missing_no_penalty_columns <- setdiff(
  required_no_penalty_columns,
  names(no_penalty_raw)
)

if (length(missing_no_penalty_columns) > 0) {
  stop(
    paste(
      "Missing no-penalty columns:",
      paste(
        missing_no_penalty_columns,
        collapse = ", "
      )
    )
  )
}

# These values repeat once for each component, so take one row
no_penalty_row <- data.frame(
  Model = "No penalty",
  MSFE = no_penalty_raw$MSFE_forecast[1],
  MAD = no_penalty_raw$MAD_forecast[1],
  MAPE = no_penalty_raw$MAPE_forecast[1],
  stringsAsFactors = FALSE
)

############################################################
# 8. Extract penalized forecast results
############################################################

penalized_table <- bic_selected[
  ,
  c(
    "Model",
    "MSFE",
    "MAD",
    "MAPE"
  )
]

############################################################
# 9. Combine no-penalty and penalized models
############################################################

table5 <- rbind(
  no_penalty_row,
  penalized_table
)

############################################################
# 10. Set exact row order used in the paper
############################################################

table5_order <- c(
  "No penalty",
  
  "Componentwise-Adaptive Group Lasso",
  "Componentwise-Group Lasso",
  "Componentwise-Group MCP",
  "Componentwise-Group SCAD",
  
  "Elementwise-Adaptive Group Lasso",
  "Elementwise-Group Lasso",
  "Elementwise-Group MCP",
  "Elementwise-Group SCAD",
  
  "Own-Other-Adaptive Group Lasso",
  "Own-Other-Group Lasso",
  "Own-Other-Group MCP",
  "Own-Other-Group SCAD"
)

table5$Model <- factor(
  table5$Model,
  levels = table5_order
)

table5 <- table5[
  order(table5$Model),
  ,
  drop = FALSE
]

table5$Model <- as.character(table5$Model)

############################################################
# 11. Check that all expected models are present
############################################################

missing_models <- setdiff(
  table5_order,
  table5$Model
)

unexpected_models <- setdiff(
  table5$Model,
  table5_order
)

if (length(missing_models) > 0) {
  warning(
    paste(
      "Missing models:",
      paste(missing_models, collapse = ", ")
    )
  )
}

if (length(unexpected_models) > 0) {
  warning(
    paste(
      "Unexpected models:",
      paste(unexpected_models, collapse = ", ")
    )
  )
}

############################################################
# 12. Round for presentation
############################################################

table5_display <- table5

table5_display$MSFE <- round(
  table5_display$MSFE,
  4
)

table5_display$MAD <- round(
  table5_display$MAD,
  4
)

table5_display$MAPE <- round(
  table5_display$MAPE,
  4
)

############################################################
# 13. Print Table 5
############################################################

cat("\nTABLE 5\n\n")

print(
  table5_display,
  row.names = FALSE
)

############################################################
# 14. Save Table 5 as CSV
############################################################

write.csv(
  table5_display,
  file.path(
    output_dir,
    "Table5_updated.csv"
  ),
  row.names = FALSE
)

############################################################
# 15. Save selected-model diagnostics
############################################################

diagnostic_columns <- intersect(
  c(
    "Model",
    "penalty",
    "HLag_structure",
    "lambda",
    "beta",
    "bic_full_data",
    "aic_full_data",
    "loglik_full_data",
    "count_non_zero_coef_full_data",
    "converged",
    "n_outer_iter",
    "runtime_sec",
    "MSFE",
    "MAD",
    "MAPE",
    "job_id"
  ),
  names(bic_selected)
)

selected_model_diagnostics <- bic_selected[
  ,
  diagnostic_columns,
  drop = FALSE
]

write.csv(
  selected_model_diagnostics,
  file.path(
    output_dir,
    "Table5_selected_model_diagnostics.csv"
  ),
  row.names = FALSE
)

############################################################
# 16. Produce LaTeX table
############################################################

latex_lines <- c(
  "\\begin{table}[htbp]",
  "\\centering",
  "\\caption{One-step-ahead forecast accuracy measures averaged across five realized volatility measures for MSFT.}",
  "\\label{tab:msft-forecast-accuracy}",
  "\\begin{tabular}{lrrr}",
  "\\hline",
  "Model & MSFE & MAD & MAPE \\\\",
  "\\hline"
)

for (i in seq_len(nrow(table5_display))) {
  
  latex_lines <- c(
    latex_lines,
    sprintf(
      "%s & %.4f & %.4f & %.4f \\\\",
      table5_display$Model[i],
      table5_display$MSFE[i],
      table5_display$MAD[i],
      table5_display$MAPE[i]
    )
  )
}

latex_lines <- c(
  latex_lines,
  "\\hline",
  "\\end{tabular}",
  "\\end{table}"
)

writeLines(
  latex_lines,
  file.path(
    output_dir,
    "Table5_updated.tex"
  )
)

cat(
  "\nSaved:\n",
  file.path(output_dir, "Table5_updated.csv"),
  "\n",
  file.path(
    output_dir,
    "Table5_selected_model_diagnostics.csv"
  ),
  "\n",
  file.path(output_dir, "Table5_updated.tex"),
  "\n"
)



########################### FIG 5 and 6 #####################################

############################################################
# SELECT BEST MODEL FOR FIGURES:
# COMPONENTWISE + GROUP LASSO
############################################################

best_model_row <- bic_selected[
  bic_selected$penalty == "group-lasso" &
    bic_selected$HLag_structure == "componentwise",
  ,
  drop = FALSE
]

if (nrow(best_model_row) != 1) {
  stop(
    paste(
      "Expected exactly one BIC-selected Componentwise",
      "Group Lasso model, but found",
      nrow(best_model_row)
    )
  )
}

print(best_model_row)

lambda_selected <- best_model_row$lambda[1]

cat(
  "\nSelected model: Componentwise-Group Lasso\n",
  "Selected lambda:", lambda_selected, "\n",
  "BIC:", best_model_row$bic_full_data[1], "\n",
  "MSFE:", best_model_row$MSFE[1], "\n",
  "MAD:", best_model_row$MAD[1], "\n",
  "MAPE:", best_model_row$MAPE[1], "\n"
)

############################################################
# REFIT THE BIC-SELECTED COMPONENTWISE GROUP LASSO MODEL
############################################################

selected_fit <- run_sparse_grid_new(
  X_train = X_train,
  X_test = X_test,
  h_step = 1,
  
  lambda_grid = lambda_selected,
  penalty_vec = "group-lasso",
  Hlag_vec = "componentwise",
  
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
  fit_unpen = fit_mle,
  
  start_from_mle = TRUE,
  trace = TRUE
)

# ============================================================
# EXTRACT SELECTED PENALIZED ESTIMATES
# ============================================================

A_selected <- selected_fit$all_fits[[1]]$fit$A_estimate
beta_selected <- selected_fit$all_fits[[1]]$fit$beta_estimate

# Compute in-sample fitted values
fitted_selected <- fitted_values_logvMEM_full(
  X = X_train,
  A_matrix = A_selected,
  beta_par = beta_selected,
  p = p
)

# Exclude the first p observations used for initialization
valid_idx <- (p + 1):nrow(X_train)

actual_values <- as.matrix(X_train[valid_idx, , drop = FALSE])
fitted_values <- as.matrix(fitted_selected[valid_idx, , drop = FALSE])

# Multiplicative residuals
multiplicative_residuals <- actual_values / fitted_values

# Optional series names
series_names <- colnames(X_train)

if (is.null(series_names)) {
  series_names <- paste0("Series ", seq_len(ncol(X_train)))
}

# ============================================================
# FIGURE 5: ACTUAL VERSUS FITTED VALUES
# ============================================================

png(
  filename = file.path(output_dir, "Figure_5_actual_vs_fitted.png"),
  width = 2400,
  height = 1800,
  res = 300
)

par(
  mfrow = c(3, 2),
  mar = c(4, 4, 3, 1),
  oma = c(0, 0, 1, 0)
)

for (j in seq_len(ncol(actual_values))) {
  
  y_range <- range(
    actual_values[, j],
    fitted_values[, j],
    finite = TRUE
  )
  
  plot(
    actual_values[, j],
    type = "l",
    col = "black",
    lwd = 1,
    ylim = y_range,
    xlab = "Time",
    ylab = "Normalized realized volatility",
    main = paste0("(", letters[j], ") ")
  )
  
  lines(
    fitted_values[, j],
    col = "green",
    lwd = 1
  )
  
  if (j == 1) {
    legend(
      "topright",
      legend = c("Actual", "Fitted"),
      col = c("black", "green"),
      lty = 1,
      lwd = 1,
      bty = "n"
    )
  }
}

# Empty sixth panel
plot.new()

dev.off()

# ============================================================
# FIGURE 6: ACF OF MULTIPLICATIVE RESIDUALS
# ============================================================

png(
  filename = file.path(output_dir, "Figure_6_residual_acf.png"),
  width = 2400,
  height = 1800,
  res = 300
)

par(
  mfrow = c(3, 2),
  mar = c(4, 4, 3, 1),
  oma = c(0, 0, 1, 0)
)

for (j in seq_len(ncol(multiplicative_residuals))) {
  
  residual_j <- multiplicative_residuals[, j]
  residual_j <- residual_j[is.finite(residual_j)]
  
  acf(
    residual_j,
    lag.max = 30,
    main = paste0("(", letters[j], ") "),
    xlab = "Lag",
    ylab = "ACF"
  )
}

# Empty sixth panel
plot.new()

dev.off()

#################################### FIG 4 ########################################################
library(lattice)

# ============================================================
# ORIGINAL SPARSITY-PLOT FUNCTION
# ============================================================

SparsityPlot <- function(B, p, k, s, m, title = NULL) {
  
  text <- c()
  
  for (i in seq_len(p)) {
    text1 <- as.expression(
      bquote(bold(A)^{"(" * .(i) * ")"})
    )
    text <- append(text, text1)
  }
  
  if (s > 0) {
    for (i in (p + 1):(p + s + 1)) {
      text1 <- as.expression(
        bquote(bold(beta)^"(" * .(i - p) * ")")
      )
      text <- append(text, text1)
    }
  }
  
  f <- function(m) {
    t(m)[, nrow(m):1, drop = FALSE]
  }
  
  rgb.palette <- colorRampPalette(
    c("white", "grey"),
    space = "Lab"
  )
  
  at <- seq(
    k / 2 + 0.5,
    p * k + 0.5,
    by = k
  )
  
  at2 <- seq(
    p * k + s / 2 + 0.5,
    p * k + s * m + 0.5,
    by = s
  )
  
  at <- c(at, at2)
  
  levelplot(
    f(abs(B)),
    col.regions = rgb.palette,
    at = c(-0.5, 0.5, 1.5),
    colorkey = NULL,
    xlab = NULL,
    ylab = NULL,
    main = list(
      label = title,
      cex = 1
    ),
    panel = function(...) {
      
      panel.levelplot(...)
      
      # Thin grid lines around all cells
      panel.abline(
        h = seq(
          1.5,
          m * s + p * k + 0.5,
          by = 1
        ),
        v = seq(
          1.5,
          by = 1,
          length = p * k + m * s
        )
      )
      
      # Boundaries between lag matrices
      lag_boundaries <- seq(
        k + 0.5,
        p * k + 0.5,
        by = k
      )
      
      panel.abline(
        v = lag_boundaries,
        lwd = 3
      )
    },
    scales = list(
      x = list(
        alternating = 1,
        labels = text,
        cex = 1,
        at = at,
        tck = c(0, 0)
      ),
      y = list(
        alternating = 0,
        tck = c(0, 0)
      )
    )
  )
}

s = 0
m = 0

# ============================================================
# EXTRACT SELECTED PENALIZED MATRIX
# ============================================================

A_penalized_estimate <-
  selected_fit$all_fits[[1]]$fit$A_estimate

# Convert coefficients into inactive/active indicators
A_penalized_estimate[
  abs(A_penalized_estimate) <= threshold
] <- 0

A_penalized_estimate[
  abs(A_penalized_estimate) > threshold
] <- 1

sum(A_penalized_estimate)

selected_fit$summary_df$count_non_zero_coef_full_data

# ============================================================
# FIGURE 4
# ============================================================

filename4 <- file.path(
  output_dir,
  "own-other_group-mcp_estimated_plot_bic.pdf"
)

pdf(
  filename4,
  width = 14,
  height = 2.7
)

print(
  SparsityPlot(
    B = A_penalized_estimate,
    p = p,
    k = d,
    s = 0,
    m = 0,
    title = NULL
  )
)

dev.off()

