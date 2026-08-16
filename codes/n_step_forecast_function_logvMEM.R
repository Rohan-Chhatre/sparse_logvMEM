
##################################
# Forecast function for log-vMEM:
##################################

forecast_logvMEM_full_data <- function(X_train,X_test,h_step,omega_estimated,A_matrix_estimated,maxlag){
  test_data <- matrix(X_test,nrow=h_step)
  par_matrix <- as.data.frame(A_matrix_estimated)
  p <- maxlag
  d <- ncol(X_train)
  omega <- omega_estimated
  
  #####################################
  # Obtain h-step ahead forecasts:
  #####################################
  n_t = nrow(X_train)
  model_data <- as.data.frame(X_train)
  
  # A matrix list:
  A_matrices <- par_matrix
  start_id <- seq(1,length(A_matrices),by=d)
  end_id <- seq(d,length(A_matrices),by=d)
  A_matrices_list <- list()

  for(i in 1:length(start_id)){
    A_matrices_list[[i]] <- A_matrices[start_id[i]:end_id[i]]
  }
  
  for(forecast_id in 1:h_step){
    indices <- (n_t+forecast_id-1):(n_t+forecast_id-p)
    model_data[n_t+forecast_id,] <- exp(omega + Reduce(f = "+", x = lapply(1:p,function(id) as.matrix(A_matrices_list[[id]]) %*% matrix(as.numeric(log(model_data[indices[id],]))))))
  }
  
  forecast_data <- model_data[(n_t+1):(n_t+h_step),]
  actual_data <- test_data
  
  df = cbind.data.frame(actual_data = t(actual_data),forecast_data = t(forecast_data))
  colnames(df) = c("actual","forecast")
  df$MSFE_i = (df$actual - df$forecast)^2
  df$MAD_i = abs(df$actual - df$forecast)
  df$MAPE_i = abs((df$actual - df$forecast)/df$actual)
  
  #################################
  # Calculate the error metrics:
  #################################
  
  # Calculate MSFE:
  (msfe <- mean(unlist(lapply(1:ncol(test_data),function(id) mean((test_data[,id] - forecast_data[,id])^2)))))
  
  # Calculate MAD:
  (mad <- mean(unlist(lapply(1:ncol(test_data),function(id) abs((test_data[,id] - forecast_data[,id]))))))
  
  # Calculate MAPE:
  (mape <- mean(unlist(lapply(1:ncol(test_data),function(id) abs((test_data[,id] - forecast_data[,id])/test_data[,id])))))
  
  model_metrics <- c(MSFE = msfe, MAD = mad,MAPE = mape)
  
  # Rteurn object:
  ret_obj = list(model_metrics = model_metrics,forecast_df = df)
  
  return(ret_obj)
  
}


# Usage of the function:
# X: is a data matrix 

# X_train <- X[1:1499,]
# X_test <- X[1500,]
# h_step <- 1
# omega_estimated <- omega_mle
# A_matrix_estimated <- A_matrix_estimated_mle
# maxlag <- 10
# 
# forecast_logvMEM_full_data(X_train,X_test,h_step,omega_estimated,A_matrix_estimated,maxlag)


  
  
