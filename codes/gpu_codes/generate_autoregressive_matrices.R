library(Matrix)
library(RSpectra)

autoregressive_matrix <- function(d, p, L, max_eigen_abs) {
  
  # Generate A matrices
  A_matrix_list <- lapply(
    seq_len(p),
    function(k) {
      matrix(
        rnorm(d^2, mean = 10, sd = 10),
        nrow = d,
        ncol = d
      )
    }
  )
  
  # ==========================================================
  # p = 1
  # ==========================================================
  if (p == 1) {
    
    # Impose HLag structure first
    A_matrix_list[[1]][1 > L] <- 0
    
    companion_matrix <- Matrix(
      A_matrix_list[[1]],
      sparse = TRUE
    )
    
    max_eigen <- Mod(
      RSpectra::eigs(
        companion_matrix,
        k = 1,
        which = "LM"
      )$values[1]
    )
    
    n_scalings <- 0L
    
    while (max_eigen > max_eigen_abs) {
      
      companion_matrix <-
        0.90 * companion_matrix
      
      max_eigen <- Mod(
        RSpectra::eigs(
          companion_matrix,
          k = 1,
          which = "LM"
        )$values[1]
      )
      
      n_scalings <- n_scalings + 1L
    }
    
    A_matrix_list <- list(
      as.matrix(companion_matrix)
    )
  }
  
  # ==========================================================
  # p > 1
  # ==========================================================
  else {
    
    # Impose HLag structure
    for (k in seq_len(p)) {
      A_matrix_list[[k]][k > L] <- 0
    }
    
    # Construct sparse companion matrix
    companion_matrix <- Matrix(
      0,
      nrow = d * p,
      ncol = d * p,
      sparse = TRUE
    )
    
    companion_matrix[1:d, ] <-
      do.call(cbind, A_matrix_list)
    
    # Lower identity block
    identity_mat <- Diagonal(
      n = d * (p - 1)
    )
    
    companion_matrix[
      (d + 1):(d * p),
      1:(d * (p - 1))
    ] <- identity_mat
    
    # Dominant eigenvalue only
    max_eigen <- Mod(
      RSpectra::eigs(
        companion_matrix,
        k = 1,
        which = "LM"
      )$values[1]
    )
    
    n_scalings <- 0L
    
    # Scale top block until stability condition is satisfied
    while (max_eigen > max_eigen_abs) {
      
      companion_matrix[1:d, ] <-
        0.90 * companion_matrix[1:d, ]
      
      max_eigen <- Mod(
        RSpectra::eigs(
          companion_matrix,
          k = 1,
          which = "LM"
        )$values[1]
      )
      
      n_scalings <- n_scalings + 1L
    }
    
    # Extract final A matrices
    A <- companion_matrix[1:d, ]
    
    A_matrix_list <- vector("list", p)
    
    for (l in seq_len(p)) {
      
      cols <- ((l - 1) * d + 1):(l * d)
      
      A_matrix_list[[l]] <-
        as.matrix(A[, cols, drop = FALSE])
    }
  }
  
  return(
    list(
      companion_matrix = companion_matrix,
      A_matrix_list = A_matrix_list,
      max_eigen = max_eigen,
      n_scalings = n_scalings
    )
  )
}

# 
# set.seed(12345)
# 
# system.time({
#   mat_test <- autoregressive_matrix(
#     d = 500,
#     p = 10,
#     L = L,
#     max_eigen_abs = 0.96
#   )
# })
# 
# mat_test$max_eigen
# mat_test$n_scalings
# dim(mat_test$companion_matrix)
# length(mat_test$A_matrix_list)
# dim(mat_test$A_matrix_list[[1]])
