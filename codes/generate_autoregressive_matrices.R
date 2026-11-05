
library(Matrix)
#set.seed(12345)

#####################################
# Generate autoregressive matrices:
#####################################

# Input:
# 1. d: time series dimension
# 2. p: maximal lag order
# 3. L: Lag matrix structure of dimension d*d

# Output:
# A sparse Matrix of class "dgCMatrix" of dimension d * dp (Companion form)
# list of A_matrices
# autoregressive_matrix <- function(d,p,L,max_eigen_abs){
#   A_matrix_list <- lapply(1:p, function(x) matrix(rnorm(n = d^2,mean=10,sd=10),nrow=d,ncol=d))
# 
#   companion_matrix <- Matrix(0,nrow=d*p,ncol = d*p)
#   for(k in 1:length(A_matrix_list)){
#       for(i in 1:d){
#         for(j in 1:d){
#           if(k> L[i,j]){
#             A_matrix_list[[k]][i,j] <- 0
#           }
#         }
#       }
#     }
#     companion_matrix[1:d,] <- do.call(cbind,A_matrix_list)
#     identity_mat <- bdiag(lapply(1:(p-1), function(x) diag(1,nrow = d,ncol = d)))
#     companion_matrix[(d+1):nrow(companion_matrix),1:ncol(identity_mat)] <- identity_mat
#     max_eigen <- max(Mod(eigen(companion_matrix)$values))
#     
#     while(max_eigen > max_eigen_abs){
#       companion_matrix[1:d,] <- 0.99*companion_matrix[1:d,]
#       identity_mat <- bdiag(lapply(1:(p-1), function(x) diag(1,nrow = d,ncol = d)))
#       companion_matrix[(d+1):nrow(companion_matrix),1:ncol(identity_mat)] <- identity_mat
#       max_eigen <- max(Mod(eigen(companion_matrix)$values))
#     }
# 
#     # Final A_matrix_list:
#     A <- companion_matrix[1:d,]
#     A_matrix_list <- list()
#     id <- seq(1,ncol(A),by=d)
#     for(l in 1:p){
#       A_matrix_list[[l]] <- as.matrix(A[,id[l]:(l*d)])
#     }
#   return(list(companion_matrix=companion_matrix,A_matrix_list=A_matrix_list))
# }

autoregressive_matrix <- function(d,p,L,max_eigen_abs){
  A_matrix_list <- lapply(1:p, function(x) matrix(rnorm(n = d^2,mean=10,sd=10),nrow=d,ncol=d))
  
  if(p==1){
    companion_matrix = A_matrix_list[[1]]
    max_eigen <- max(Mod(eigen(companion_matrix)$values))
    
    while(max_eigen > max_eigen_abs){
      companion_matrix[1:d,] <- 0.99*companion_matrix[1:d,]
      max_eigen <- max(Mod(eigen(companion_matrix)$values))
    }
    
    # Final A_matrix_list:
    A <- companion_matrix[1:d,]
    A_matrix_list <- list()
    id <- seq(1,ncol(A),by=d)
    for(l in 1:p){
      A_matrix_list[[l]] <- as.matrix(A[,id[l]:(l*d)])
    }
    
    for(k in 1:length(A_matrix_list)){
      for(i in 1:d){
        for(j in 1:d){
          if(k> L[i,j]){
            A_matrix_list[[k]][i,j] <- 0
          }
        }
      }
    }
  }
  else{
    companion_matrix <- Matrix(0,nrow=d*p,ncol = d*p)
    for(k in 1:length(A_matrix_list)){
      for(i in 1:d){
        for(j in 1:d){
          if(k> L[i,j]){
            A_matrix_list[[k]][i,j] <- 0
          }
        }
      }
    }
    companion_matrix[1:d,] <- do.call(cbind,A_matrix_list)
    identity_mat <- bdiag(lapply(1:(p-1), function(x) diag(1,nrow = d,ncol = d)))
    companion_matrix[(d+1):nrow(companion_matrix),1:ncol(identity_mat)] <- identity_mat
    max_eigen <- max(Mod(eigen(companion_matrix)$values))
    
    while(max_eigen > max_eigen_abs){
      companion_matrix[1:d,] <- 0.99*companion_matrix[1:d,]
      identity_mat <- bdiag(lapply(1:(p-1), function(x) diag(1,nrow = d,ncol = d)))
      companion_matrix[(d+1):nrow(companion_matrix),1:ncol(identity_mat)] <- identity_mat
      max_eigen <- max(Mod(eigen(companion_matrix)$values))
    }
    
    # Final A_matrix_list:
    A <- companion_matrix[1:d,]
    A_matrix_list <- list()
    id <- seq(1,ncol(A),by=d)
    for(l in 1:p){
      A_matrix_list[[l]] <- as.matrix(A[,id[l]:(l*d)])
    }
  }
  return(list(companion_matrix=companion_matrix,A_matrix_list=A_matrix_list))
}


# Working example:
# d <- 2
# p <- 3
# L <- matrix(c(rep(2,2),rep(1,2)),nrow=d,ncol=d,byrow = TRUE) # component-wise Hlag structure
# max_eigen_abs <- 0.95
# mat <- autoregressive_matrix(d = d,p = p,L = L,max_eigen_abs = max_eigen_abs)
# 
# round(mat$companion_matrix[1:d,],2)
