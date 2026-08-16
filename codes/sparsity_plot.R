
#######################################################
# Adapted from Nicholson et al. paper and Github repo:
#######################################################

library(lattice)

SparsityPlot <- function (B, p, k,s,m, title = NULL) 
{
  text <- c()
  for (i in 1:p) {
    text1 <- as.expression(bquote(bold(A)^(.(i))))
    text <- append(text, text1)
  }
  ## text <- c()
  if(s>0){
    for (i in (p+1):(p+s+1)) {
      text1 <- as.expression(bquote(bold(beta)^(.(i-p))))
      text <- append(text, text1)
    }
  }
  f <- function(m) t(m)[, nrow(m):1]
  
  rgb.palette <- colorRampPalette(c("white", "grey" ),space = "Lab")
  ## rgb.palette <- colorRampPalette(c("white", "blue"), space = "Lab")
  at <- seq(k/2 + 0.5, p * (k)+ 0.5, by = k)
  at2 <- seq(p*k+s/2+.5,p*k+s*m+.5,by=s)
  at <- c(at,at2)
  se2 = seq(1.75, by = k, length = k)
  L2 <- levelplot(f(abs(B)), col.regions = rgb.palette, colorkey = NULL, 
                  xlab = NULL, ylab = NULL, main = list(label = title, 
                                                        cex = 1), panel = function(...) {
                                                          panel.levelplot(...)
                                                          panel.abline(a = NULL, b = 1, h = seq(1.5, m*s+p* k + 
                                                                                                  0.5, by = 1), v = seq(1.5, by = 1, length = p * 
                                                                                                                          k+m*s))
                                                          bl1 <- seq(k + 0.5, p * 
                                                                       k + 0.5, by = k)
                                                          bl2 <- seq(p*k + 0.5, p * 
                                                                       k + 0.5+s*m, by = m)
                                                          b1 <- c(bl1,bl2)
                                                          panel.abline(a = NULL, b = 1, v = p*k+.5, lwd = 7)
                                                          panel.abline(a = NULL, b = 1, v = b1, lwd = 3)
                                                        }, scales = list(x = list(alternating = 1, labels = text, 
                                                                                  cex = 1, at = at, tck = c(0, 0)), y = list(alternating = 0, 
                                                                                                                             tck = c(0, 0))))
  return(L2)
}

# B1 <- matrix(rep(1,57)*rbinom(57,1,.6),nrow=3,ncol=19)
# B2 <-matrix(0,nrow=3,ncol=19)
# B2[,1:3] <- 1
# B2[,10:12] <- 1
# B2[,16] <- 1
# B2[,19] <- 1
# B3 <-matrix(0,nrow=3,ncol=19)
# diag(B3[,1:3])<- 1
# B3[,10:12] <- 1
# diag(B3[,10:12])<- 0
# B2[,16] <- 1
# B2[,19] <- 1
# B4 <-matrix(0,nrow=3,ncol=19)
# B4[,1:3] <- rep(1,9)*rbinom(9,1,.6)
# B4[,10:12] <- rep(1,9)*rbinom(9,1,.4)
# B4[,18] <- c(1,0,1)
# p=5;k=3
# 
# 
# 
# SparsityPlot(B1,p,k,m=2,s=2,title="Basic VARX-L")

