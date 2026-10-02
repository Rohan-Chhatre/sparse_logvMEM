library(Matrix)
library(RSpectra)

#####################################
# Generate autoregressive matrices:
#####################################

autoregressive_matrix_own_other_diag_dom <- function(
    d,
    p,
    L,
    max_eigen_abs
) {
  
  # ----------------------------------------------------------
  # 1. Generate coefficient matrices
  # ----------------------------------------------------------
  
  A_matrix_list <- lapply(
    seq_len(p),
    function(k) {
      matrix(
        rnorm(
          n = d^2,
          mean = 10,
          sd = 10
        ),
        nrow = d,
        ncol = d
      )
    }
  )
  
  # ----------------------------------------------------------
  # 2. Diagonal dominance
  # ----------------------------------------------------------
  
  for (k in seq_len(p)) {
    
    max_val <- apply(
      abs(A_matrix_list[[k]]),
      1,
      max
    )
    
    diag(A_matrix_list[[k]]) <- max_val + 1
  }
  
  # ----------------------------------------------------------
  # 3. Impose HLag structure
  # ----------------------------------------------------------
  # Equivalent to the original i-j-k loops, but vectorized.
  
  for (k in seq_len(p)) {
    A_matrix_list[[k]][k > L] <- 0
  }
  
  # ----------------------------------------------------------
  # 4. Construct companion matrix
  # ----------------------------------------------------------
  
  companion_matrix <- Matrix(
    0,
    nrow = d * p,
    ncol = d * p,
    sparse = TRUE
  )
  
  companion_matrix[1:d, ] <- do.call(
    cbind,
    A_matrix_list
  )
  
  identity_mat <- Diagonal(
    n = d * (p - 1)
  )
  
  companion_matrix[
    (d + 1):(d * p),
    1:(d * (p - 1))
  ] <- identity_mat
  
  # ----------------------------------------------------------
  # 5. Calculate dominant eigenvalue only
  # ----------------------------------------------------------
  
  max_eigen <- Mod(
    RSpectra::eigs(
      companion_matrix,
      k = 1,
      which = "LM"
    )$values[1]
  )
  
  # ----------------------------------------------------------
  # 6. Scale until stable
  # ----------------------------------------------------------
  
  n_scalings <- 0L
  
  while (max_eigen > max_eigen_abs) {
    
    # Preserve original scaling rule
    companion_matrix[1:d, ] <-
      0.9 * companion_matrix[1:d, ]
    
    # No need to reconstruct identity_mat:
    # those rows have not changed.
    
    max_eigen <- Mod(
      RSpectra::eigs(
        companion_matrix,
        k = 1,
        which = "LM"
      )$values[1]
    )
    
    n_scalings <- n_scalings + 1L
  }
  
  # ----------------------------------------------------------
  # 7. Extract final A matrices
  # ----------------------------------------------------------
  
  A <- companion_matrix[1:d, ]
  
  A_matrix_list <- vector(
    "list",
    p
  )
  
  for (l in seq_len(p)) {
    
    cols <- ((l - 1) * d + 1):(l * d)
    
    A_matrix_list[[l]] <- as.matrix(
      A[, cols, drop = FALSE]
    )
  }
  
  # ----------------------------------------------------------
  # 8. Return
  # ----------------------------------------------------------
  
  return(
    list(
      companion_matrix = companion_matrix,
      A_matrix_list = A_matrix_list,
      max_eigen = max_eigen,
      n_scalings = n_scalings
    )
  )
}



# system.time({
#   
#   mat_test <- autoregressive_matrix_own_other_diag_dom(
#     d = 500,
#     p = 10,
#     L = L,
#     max_eigen_abs = 0.96
#   )
#   
# })
# 
# mat_test$max_eigen
# mat_test$n_scalings

# system.time({
#   mat_test <- autoregressive_matrix_own_other_diag_dom(
#     d = 500,
#     p = 10,
#     L = L,
#     max_eigen_abs = 0.96
#   )
# })
# 
# mat_test$max_eigen
# mat_test$n_scalings
