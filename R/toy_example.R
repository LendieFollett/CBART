rm(list = ls())


library(Rcpp)
install.packages("devtools")#install devtools for function install_github
library(devtools)
install_github("LendieFollett/CBART")
library(CBART)
library(dplyr)
library(ggplot2)
beta <-c(-2, 4)
rho <- .7
sig <-1
p <-1
n <- 1000
ntest <- 500

train_idx <- 1:(n)
test_idx <- (n+1):(n + ntest)

set.seed(352833 )
x <- matrix(rbeta((n + ntest)*p, 1,1), ncol = p, byrow = FALSE)
#true_lambda <- exp(true_beta0)
true_f <- rep(NA, n)
true_f2 <- rep(NA, n)
true_u <- rep(NA, n)

for (i in 1:(n + ntest)){
  set.seed(2388520 + i)
  true_u[i] <- rexp(1, rate = 1/exp(beta[1] + beta[2]*x[i,1]))
  set.seed(2352 + i )
  epsilon <- rnorm(1, 0, sig)
  true_f[i] <- sin(pi*(x[i,1]-0.5))
  true_f2[i] <- true_f[i] + epsilon
}

set.seed(23589500)
y <- true_f2  - true_u*rbinom((n + ntest), size = 1, prob = rho)

x_train <- x[train_idx]%>%as.matrix()
x_test <- x[-train_idx]%>%as.matrix()
y_train <- y[train_idx]
y_test <-y[-train_idx]

temp <- data.frame(y = y_train, x=x_train,x2 = x_train^2,x3 = x_train^3)
#start with OLR
olr <- lm(y ~ x, data = temp)
olr_pred <- predict(olr, data.frame(x=x_train,x2 = x_train^2,x3 = x_train^3))
olr_resid <- olr_pred- y_test
temp <- data.frame(y = log(pmax(olr_resid,0)+.2),x =x_train )
lambda_start <- lm(y ~., data= temp) %>%coef()
lambda_sd <- ((lm(y ~., data= temp) %>%vcov())%>%diag()%>%sqrt())*2

set.seed(367231 )  ## MCMC posterior sampling: set seed for reproducibility
bf = cbart(x_train,y[train_idx],nskip=100,
                 sigdf=3,
                 ntree = 100,
                 sigquant = .9,
                 sigest = sqrt(mean(olr_resid^2))/2, #half the standard deviation
                 ndpost=500,
                 x.test = x_test,
                 printevery=100L,
                 sparse = TRUE,
                 ux.train = cbind(1, x_train), #training data for asymmetric betas
                 ux.test =cbind(1, x_test),
                 lambda_prop_sd = lambda_sd,
                 lambda_start = lambda_start,
                 gamma_prop_sd = 1) #testing data for asymmetric betas


