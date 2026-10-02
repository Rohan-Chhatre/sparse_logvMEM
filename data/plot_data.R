# ============================================================
# MSFT REALIZED VOLATILITY:
# TRAIN + TEST WITH DATE AXIS AND TEST-SAMPLE DIVIDER
# ============================================================


# ============================================================
# 1. FILE PATHS
# ============================================================

train_path <- "/Users/rohan/Documents/Rohan/Finance paper/Sparse-Logarithmic-Vector-Multiplicative-Error-Model-with-Multivariate-Gamma-Errors-develop_RCodes/data/realized_volatility_measures_MSFT_train.csv"

test_path <- "/Users/rohan/Documents/Rohan/Finance paper/Sparse-Logarithmic-Vector-Multiplicative-Error-Model-with-Multivariate-Gamma-Errors-develop_RCodes/data/realized_volatility_measures_MSFT_test.csv"


# ============================================================
# 2. READ DATA
# ============================================================

train_data <- read.csv(
  train_path,
  stringsAsFactors = FALSE
)

test_data <- read.csv(
  test_path,
  stringsAsFactors = FALSE
)


# ============================================================
# 3. SELECT THE FIVE VOLATILITY MEASURES BY COLUMN NAME
# ============================================================

vol_cols <- c(
  "rv_med_5min",
  "rvKernel_mth_5min",
  "rbp_cov_5min",
  "rv_med_10min",
  "rvKernel_mth_10min"
)

# Select by name so it does not matter that the columns are
# in different positions in the two files
train_plot <- train_data[, c("DT", vol_cols)]

test_plot <- test_data[, c("DT", vol_cols)]


# ============================================================
# 4. CONVERT DATES
# ============================================================

# Training file:
# example "1/4/2010" = MM/DD/YYYY
train_plot$DT <- as.Date(
  train_plot$DT,
  format = "%m/%d/%Y"
)

# Test file:
# example "2015-11-06" = YYYY-MM-DD
test_plot$DT <- as.Date(
  test_plot$DT,
  format = "%Y-%m-%d"
)


# ============================================================
# 5. CHECK DATE CONVERSION
# ============================================================

cat("\nFirst training dates:\n")
print(head(train_plot$DT))

cat("\nFirst test dates:\n")
print(head(test_plot$DT))

cat("\nNumber of missing training dates:",
    sum(is.na(train_plot$DT)), "\n")

cat("Number of missing test dates:",
    sum(is.na(test_plot$DT)), "\n")


# ============================================================
# 6. MAKE SURE VOLATILITY COLUMNS ARE NUMERIC
# ============================================================

for (v in vol_cols) {
  
  train_plot[[v]] <- as.numeric(train_plot[[v]])
  test_plot[[v]]  <- as.numeric(test_plot[[v]])
  
}


# ============================================================
# 7. SORT EACH DATASET CHRONOLOGICALLY
# ============================================================

train_plot <- train_plot[
  order(train_plot$DT),
]

test_plot <- test_plot[
  order(test_plot$DT),
]


# ============================================================
# 8. COMBINE TRAIN FIRST, THEN TEST
# ============================================================

full_data <- rbind(
  train_plot,
  test_plot
)

# Beginning of test sample
test_start_date <- min(
  test_plot$DT,
  na.rm = TRUE
)


# ============================================================
# 9. PRINT SAMPLE INFORMATION
# ============================================================

cat("\n-----------------------------------\n")
cat("MSFT DATA SUMMARY\n")
cat("-----------------------------------\n")

cat(
  "Training observations:",
  nrow(train_plot),
  "\n"
)

cat(
  "Test observations:",
  nrow(test_plot),
  "\n"
)

cat(
  "Total observations:",
  nrow(full_data),
  "\n"
)

cat(
  "Training period:",
  format(min(train_plot$DT, na.rm = TRUE), "%Y-%m-%d"),
  "to",
  format(max(train_plot$DT, na.rm = TRUE), "%Y-%m-%d"),
  "\n"
)

cat(
  "Test period:",
  format(min(test_plot$DT, na.rm = TRUE), "%Y-%m-%d"),
  "to",
  format(max(test_plot$DT, na.rm = TRUE), "%Y-%m-%d"),
  "\n"
)

cat(
  "Test sample begins:",
  format(test_start_date, "%Y-%m-%d"),
  "\n"
)

cat("-----------------------------------\n")


# ============================================================
# 10. PANEL LABELS
# ============================================================

panel_labels <- c(
  "(a)",
  "(b)",
  "(c)",
  "(d)",
  "(e)"
)


# ============================================================
# 11. CREATE YEAR TICKS FOR X AXIS
# ============================================================

first_year <- as.integer(
  format(
    min(full_data$DT, na.rm = TRUE),
    "%Y"
  )
)

last_year <- as.integer(
  format(
    max(full_data$DT, na.rm = TRUE),
    "%Y"
  )
)

date_ticks <- as.Date(
  paste0(
    seq(first_year, last_year),
    "-01-01"
  )
)


# ============================================================
# 12. PLOT ON SCREEN
# ============================================================

par(
  mfrow = c(3, 2),
  mar = c(4, 4, 3, 1),
  oma = c(1, 1, 1, 1)
)

for (j in seq_along(vol_cols)) {
  
  y <- full_data[[vol_cols[j]]]
  
  plot(
    full_data$DT,
    y,
    type = "l",
    col = "black",
    lwd = 0.8,
    xlab = "Date",
    ylab = "Normalized realized volatility",
    main = panel_labels[j],
    xaxt = "n"
  )
  
  
  # ----------------------------------------------------------
  # X axis: years
  # ----------------------------------------------------------
  
  axis(
    side = 1,
    at = date_ticks,
    labels = format(date_ticks, "%Y"),
    las = 1
  )
  
  
  # ----------------------------------------------------------
  # Vertical dotted line at beginning of test sample
  # ----------------------------------------------------------
  
  abline(
    v = test_start_date,
    lty = 3,
    lwd = 1.5
  )
  
}

# Blank sixth panel
plot.new()


# ============================================================
# 13. SAVE HIGH-RESOLUTION PNG
# ============================================================

output_file <- "/Users/rohan/Documents/Rohan/Finance paper/Sparse-Logarithmic-Vector-Multiplicative-Error-Model-with-Multivariate-Gamma-Errors-develop_RCodes/Figure_5_MSFT_train_test.png"

png(
  filename = output_file,
  width = 2400,
  height = 1800,
  res = 300
)

par(
  mfrow = c(3, 2),
  mar = c(4, 4, 3, 1),
  oma = c(1, 1, 1, 1)
)

for (j in seq_along(vol_cols)) {
  
  y <- full_data[[vol_cols[j]]]
  
  plot(
    full_data$DT,
    y,
    type = "l",
    col = "black",
    lwd = 0.8,
    xlab = "Date",
    ylab = "Normalized realized volatility",
    main = panel_labels[j],
    xaxt = "n"
  )
  
  
  # X axis
  axis(
    side = 1,
    at = date_ticks,
    labels = format(date_ticks, "%Y"),
    las = 1
  )
  
  
  # Dotted train/test separator
  abline(
    v = test_start_date,
    lty = 3,
    lwd = 1.5,
    col='red'
  )
  
}

# Blank sixth panel
plot.new()

dev.off()


cat(
  "\nFigure saved to:\n",
  output_file,
  "\n"
)