
## BART: Bayesian Additive Regression Trees
## Copyright (C) 2017 Robert McCulloch and Rodney Sparapani

## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program; if not, a copy is available at
## https://www.R-project.org/Licenses/GPL-2

cbart=function(
x.train, y.train, x.test=matrix(0.0,0,0),
ux.train, ux.test, #LRF
sparse=FALSE, theta=0, omega=1,
a=0.5, b=1, augment=FALSE, rho=NULL,
xinfo=matrix(0.0,0,0), usequants=FALSE,
cont=FALSE, rm.const=TRUE,
sigest=NA, sigdf=3, sigquant=.90,
k=2.0, power=2.0, base=.95,
sigmaf=NA, lambda=NA,
fmean=mean(y.train),
w=rep(1,length(y.train)),
ntree=200L, numcut=100L,
ndpost=1000L, nskip=100L, keepevery=1L,
nkeeptrain=ndpost, nkeeptest=ndpost,
nkeeptestmean=ndpost, nkeeptreedraws=ndpost,
printevery=100L, transposed=FALSE,
lambda_prop_sd=NA, #proposal standard deviation for RW on lambda
#lambda_prior_sd = 2.5,
lambda_start=NA,
gamma_prop_sd=.25
){

#--------------------------------------------------
#data
n = length(y.train)

if(!transposed) {
    temp = bartModelMatrix(x.train, numcut, usequants=usequants,
                           cont=cont, xinfo=xinfo, rm.const=rm.const)
    x.train = t(temp$X)
    numcut = temp$numcut
    xinfo = temp$xinfo
    if(length(x.test)>0) {
            x.test = bartModelMatrix(x.test)
            x.test = t(x.test[ , temp$rm.const])
    }
    rm.const <- temp$rm.const
    grp <- temp$grp
    rm(temp)
}
else {
    rm.const <- NULL
    grp <- NULL
}

if(n!=ncol(x.train))
    stop('The length of y.train and the number of rows in x.train must be identical')

p = nrow(x.train)
np = ncol(x.test)
up = ncol(ux.train) #LRF

if(length(rho)==0) rho=p
if(length(rm.const)==0) rm.const <- 1:p
if(length(grp)==0) grp <- 1:p

##if(p>1 & length(numcut)==1) numcut=rep(numcut, p)

y.train = y.train-fmean
#--------------------------------------------------
#set nkeeps for thinning
if((nkeeptrain!=0) & ((ndpost %% nkeeptrain) != 0)) {
   nkeeptrain=ndpost
   cat('*****nkeeptrain set to ndpost\n')
}
if((nkeeptest!=0) & ((ndpost %% nkeeptest) != 0)) {
   nkeeptest=ndpost
   cat('*****nkeeptest set to ndpost\n')
}
if((nkeeptestmean!=0) & ((ndpost %% nkeeptestmean) != 0)) {
   nkeeptestmean=ndpost
   cat('*****nkeeptestmean set to ndpost\n')
}
if((nkeeptreedraws!=0) & ((ndpost %% nkeeptreedraws) != 0)) {
   nkeeptreedraws=ndpost
   cat('*****nkeeptreedraws set to ndpost\n')
}
#--------------------------------------------------
#prior
nu=sigdf
if(is.na(lambda)) {
   if(is.na(sigest)) {
      if(p < n) {
         df = data.frame(t(x.train),y.train)
         lmf = lm(y.train~.,df)
         sigest = summary(lmf)$sigma/2
      } else {
         sigest = sd(y.train)/2
      }
   }
   qchi = qchisq(1.0-sigquant,nu)
   lambda = (sigest*sigest*qchi)/nu #lambda parameter for sigma prior

} else {
   sigest=sqrt(lambda)/2
}

if(is.na(sigmaf)) {
   tau=(max(y.train)-min(y.train))/(2*k*sqrt(ntree))
} else {
   tau = sigmaf/sqrt(ntree)
}

if(any(is.na(lambda_start))){
  lambda_start = rep(0, ncol(ux.train))
}

if(any(is.na(lambda_prop_sd))){
  lambda_prop_sd = rep(.5, ncol(ux.train))
}

#--------------------------------------------------
ptm <- proc.time()
#call
res = cbart_cpp(
            n,  #number of observations in training data
            p,  #dimension of x
            up,  #dimension of x for scale of us (LRF)
            np, #number of observations in test data
            x.train,   #pxn training data x
            y.train,   #pxn training data x
            x.test,   #p*np test data x
            ux.train , #up*n train for u scale
            ux.test, #up*np test for u scale
            ntree,
            numcut,
            ndpost*keepevery,
            nskip,
            power,
            base,
            tau,
            nu,
            lambda,
            sigest,
           # u_asym_est,
            w,
            sparse,
            theta,
            omega,
            grp,
            a,
            b,
            rho,
            augment,
            nkeeptrain,
            nkeeptest,
            nkeeptestmean,
            nkeeptreedraws,
            printevery,
            xinfo,
            lambda_prop_sd,
            lambda_start,
           gamma_prop_sd
)

res$proc.time <- proc.time()-ptm

res$mu = fmean
res$cap.train = res$cap.train + fmean
res$potential.train.mean = res$potential.train.mean+fmean
res$potential.train =      res$potential.train     +fmean

res$cap.test = res$cap.test + fmean
res$potential.test.mean = res$potential.test.mean+fmean
res$potential.test = res$potential.test+fmean
if(nkeeptreedraws>0)
    names(res$treedraws$cutpoints) = dimnames(x.train)[[1]]
    dimnames(res$varcount)[[2]] = as.list(dimnames(x.train)[[1]])
    dimnames(res$varprob)[[2]] = as.list(dimnames(x.train)[[1]])
##res$nkeeptreedraws=nkeeptreedraws
    res$varcount.mean <- apply(res$varcount, 2, mean)
    res$varprob.mean <- apply(res$varprob, 2, mean)
    res$rm.const <- rm.const
#attr(res, 'class') <- 'cbart'
return(res)
}
