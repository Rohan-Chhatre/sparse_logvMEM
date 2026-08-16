# Sparsity plot of the estimated final matrix based on BIC:
A_penalized_estimate = final_estimate_df_bic[,paste("A",1:(p*d),sep="_")]
A_penalized_estimate[which(abs(A_penalized_estimate)<=threshold,arr.ind = T)]=0
A_penalized_estimate[which(abs(A_penalized_estimate)>threshold,arr.ind = T)]=1

filename2 = paste(paste(penalty,Hlag_structure,"estimated_plot_bic",sep="_"),"pdf",sep=".")
pdf(filename2)
SparsityPlot(B = A_penalized_estimate,p = p,k = d,s = 2,m = 2,title = filename2)
dev.off()

# Sparsity plot of the estimated final matrix based on MSFE:
A_penalized_estimate = final_estimate_df_msfe[,paste("A",1:(p*d),sep="_")]
A_penalized_estimate[which(abs(A_penalized_estimate)<=threshold,arr.ind = T)]=0
A_penalized_estimate[which(abs(A_penalized_estimate)>threshold,arr.ind = T)]=1

filename3 = paste(paste(penalty,Hlag_structure,"estimated_plot_MSFE",sep="_"),"pdf",sep=".")
pdf(filename3)
SparsityPlot(B = A_penalized_estimate,p = p,k = d,s = 2,m = 2,title = filename3)
dev.off()
