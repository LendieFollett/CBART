/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017 Robert McCulloch and Rodney Sparapani
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/GPL-2
 */

#include <Rcpp.h>
using namespace Rcpp;

#include "tree.h"
#include "treefuns.h"
#include "info.h"
#include "bartfuns.h"
#include "bd.h"
#include "bart.h"
#include "heterbart.h"

#ifndef NoRcpp

#define TRDRAW(a, b) trdraw(a, b)
#define TEDRAW(a, b) tedraw(a, b)
#define TECAPDRAW(a, b) tecapdraw(a, b)
#define CAPDRAW(a, b) capdraw(a, b)
#define U_ASYM_DRAW(a, b) u_asym_draw(a, b)
#define U_LAMBDA_DRAW(a, b) u_lambda_draw(a, b)


// [[Rcpp::export]]
RcppExport SEXP cbart_cpp(
   SEXP in_,            //number of observations in training data
   SEXP ip_,		//dimension of x
   SEXP iup_,		//dimension of x in scale of u's (LRF)
   SEXP inp_,		//number of observations in test data
   SEXP ix_,		//x, train,  pxn (transposed so rows are contiguous in memory)
   SEXP iy_,		//y, train,  nx1
   SEXP ixp_,		//x, test, pxnp (transposed so rows are contiguous in memory)
   SEXP iux_,    //x, train, u scale
   SEXP iuxp_, //x, test, u scale
   SEXP im_,		//number of trees
   SEXP inc_,		//number of cut points
   SEXP ind_,		//number of kept draws (except for thinnning ..)
   SEXP iburn_,		//number of burn-in draws skipped
   SEXP ipower_,
   SEXP ibase_,
   SEXP itau_,
   SEXP inu_,
   SEXP ilambda_,
   SEXP isigest_,
   SEXP iw_,
   SEXP idart_,
   SEXP itheta_,
   SEXP iomega_,
   SEXP igrp_,
   SEXP ia_,
   SEXP ib_,
   SEXP irho_,
   SEXP iaug_,
   SEXP inkeeptrain_,
   SEXP inkeeptest_,
   SEXP inkeeptestme_,
   SEXP inkeeptreedraws_,
   SEXP inprintevery_,
   SEXP Xinfo_,
  SEXP ilambda_prop_sd_,
  SEXP ilambda_start_,
  SEXP gamma_prop_sd_
)
{

   //--------------------------------------------------
   //process args
   size_t n = Rcpp::as<int>(in_);
   size_t p = Rcpp::as<int>(ip_);
   size_t up = Rcpp::as<int>(iup_);
   size_t np = Rcpp::as<int>(inp_);
   Rcpp::NumericVector  xv(ix_);
   double *ix = &xv[0];
   //Rcpp::NumericVector  uxv(_iux); //lrf
   //double *iux = &uxv[0];//lrf
   Rcpp::NumericVector  yv(iy_);
   double *iy = &yv[0];
   Rcpp::NumericVector  xpv(ixp_);
   double *ixp = &xpv[0];
   //Rcpp::NumericVector  uxpv(_iuxp);//lrf
   //double *iuxp = &uxpv[0];
   size_t m = Rcpp::as<int>(im_);
   Rcpp::IntegerVector nc_(inc_);
   int *numcut = &nc_[0];
   //size_t nc = Rcpp::as<int>(_inc);
   size_t nd = Rcpp::as<int>(ind_);
   size_t burn = Rcpp::as<int>(iburn_);
   double mybeta = Rcpp::as<double>(ipower_);
   double gamma_prop_sd = Rcpp::as<double>(gamma_prop_sd_);
   //double lambda_prior_sd = Rcpp::as<double>(_lambda_prior_sd);
   double alpha = Rcpp::as<double>(ibase_);
   double tau = Rcpp::as<double>(itau_);
   double nu = Rcpp::as<double>(inu_);
   double lambda = Rcpp::as<double>(ilambda_);
   double sigma=Rcpp::as<double>(isigest_);
   //std::vector<double>u_asym(_iu_asym_est);
   //double u_asym=Rcpp::as<double>(_iu_asym_est);
  // Rcpp::NumericVector  u_asym(_iu_asym_est);
   //double *iu_asym_est = &u_asym[0];

   Rcpp::NumericVector  wv(iw_);
   double *iw = &wv[0];
   Rcpp::NumericVector lambda_start(ilambda_start_);
   double *ilambda_start = &lambda_start[0];

   Rcpp::NumericVector lambda_prop_sd(ilambda_prop_sd_);
   double *ilambda_prop_sd = &lambda_prop_sd[0];

  // double lambda_prop_sd = Rcpp::as<double>(_lambda_prop_sd);

   bool dart;
   if(Rcpp::as<int>(idart_)==1) dart=true;
   else dart=false;
   double a = Rcpp::as<double>(ia_);
   double b = Rcpp::as<double>(ib_);
   double rho = Rcpp::as<double>(irho_);
   bool aug;
   if(Rcpp::as<int>(iaug_)==1) aug=true;
   else aug=false;
   double theta = Rcpp::as<double>(itheta_);
   double omega = Rcpp::as<double>(iomega_);
   Rcpp::IntegerVector grp_(igrp_);
   int *grp = &grp_[0];
   size_t nkeeptrain = Rcpp::as<int>(inkeeptrain_);
   size_t nkeeptest = Rcpp::as<int>(inkeeptest_);
   size_t nkeeptestme = Rcpp::as<int>(inkeeptestme_);
   size_t nkeeptreedraws = Rcpp::as<int>(inkeeptreedraws_);
   size_t printevery = Rcpp::as<int>(inprintevery_);
//   int treesaslists = Rcpp::as<int>(_treesaslists);
   Rcpp::NumericMatrix Xinfo(Xinfo_);
   Rcpp::NumericMatrix iux(iux_);
   Rcpp::NumericMatrix iuxp(iuxp_);
//   Rcpp::IntegerMatrix varcount(nkeeptreedraws, p);

   //return data structures (using Rcpp)
   Rcpp::NumericVector trmean(n); //train
   Rcpp::NumericVector temean(np);
   Rcpp::NumericVector sdraw(nd+burn);
   Rcpp::NumericVector rhodraw(nd+burn);
   Rcpp::NumericMatrix u_lambda_draw(nkeeptrain,up); //LRF
   Rcpp::NumericMatrix u_asym_draw(nkeeptrain,n);
   Rcpp::NumericMatrix trdraw(nkeeptrain,n);
   Rcpp::NumericMatrix capdraw(nkeeptrain,n);
   Rcpp::NumericMatrix tedraw(nkeeptest,np);
   Rcpp::NumericMatrix tecapdraw(nkeeptest,np);
//   Rcpp::List list_of_lists(nkeeptreedraws*treesaslists);
   Rcpp::NumericMatrix varprb(nkeeptreedraws,p);
   Rcpp::IntegerMatrix varcnt(nkeeptreedraws,p);

   //random number generation
   arn gen;

   heterbart bm(m);

   if(Xinfo.size()>0) {
     xinfo _xi;
     _xi.resize(p);
     for(size_t i=0;i<p;i++) {
       _xi[i].resize(numcut[i]);
       //Rcpp::IntegerVector cutpts(Xinfo[i]);
       for(size_t j=0;j<numcut[i];j++) _xi[i][j]=Xinfo(i, j);
     }
     bm.setxinfo(_xi);
   }
#else

#define TRDRAW(a, b) trdraw[a][b]
#define TEDRAW(a, b) tedraw[a][b]
#define TECAPDRAW(a, b) tecapdraw[a][b]
#define CAPDRAW(a, b) capdraw[a][b]
#define U_ASYM_DRAW(a, b) u_asym_draw[a][b]

void cbart_cpp(
   size_t n,            //number of observations in training data
   size_t p,		//dimension of x
   size_t np,		//number of observations in test data
   size_t up,
   double* ix,		//x, train,  pxn (transposed so rows are contiguous in memory)
   double* iy,		//y, train,  nx1
   double* ixp,		//x, test, pxnp (transposed so rows are contiguous in memory)
   double* iux, //LRF
   double* iuxp,//LRF
   size_t m,		//number of trees
   int* numcut,		//number of cut points
   size_t nd,		//number of kept draws (except for thinnning ..)
   size_t burn,		//number of burn-in draws skipped
   double mybeta,
   double alpha,
   double tau,
   double nu,
   double lambda,
   double sigma,
   //double u_asym,
   double* iw,
   bool dart,
   double theta,
   double omega,
   int *grp,
   double a,
   double b,
   double rho,
   bool aug,
   size_t nkeeptrain,
   size_t nkeeptest,
   size_t nkeeptestme,
   size_t nkeeptreedraws,
   size_t printevery,
//   int treesaslists,
   unsigned int n1, // additional parameters needed to call from C++
   unsigned int n2,
   double* trmean,
   double* temean,
   double* sdraw,
   double* rhodraw,
   double* _trdraw,
   double* _tedraw,
   double* _tecapdraw,
   double* _capdraw,
   double* _u_asym_draw,
   double* _u_lambda_draw
)
{



   //return data structures (using C++)
   std::vector<double*> u_asym_draw(nkeeptrain);
  std::vector<double*> u_lambda_draw(nkeeptrain);
   std::vector<double*> trdraw(nkeeptrain);
   std::vector<double*> tedraw(nkeeptest);
   std::vector<double*> tecapdraw(nkeeptest);
   std::vector<double*> capdraw(nkeeptrain);

   for(size_t i=0; i<nkeeptrain; ++i) trdraw[i]=&_trdraw[i*n];
   for(size_t i=0; i<nkeeptest; ++i) tedraw[i]=&_tedraw[i*np];
   for(size_t i=0; i<nkeeptest; ++i) tecapdraw[i]=&_tecapdraw[i*np];
   for(size_t i=0; i<nkeeptrain; ++i) capdraw[i]=&_capdraw[i*n];
   for(size_t i=0; i<nkeeptrain; ++i) u_asym_draw[i]=&_u_asym_draw[i*n];
   for(size_t i=0; i<nkeeptrain; ++i) u_lambda_draw[i]=&_u_lambda_draw[i*n];


   std::vector< std::vector<size_t> > varcnt;
   std::vector< std::vector<double> > varprb;

   //random number generation
   arn gen(n1, n2);

   heterbart bm(m);
#endif

   for(size_t i=0;i<n;i++) trmean[i]=0.0;
   for(size_t i=0;i<np;i++) temean[i]=0.0;

   printf("*****Into main of wbart\n");
   //-----------------------------------------------------------

   size_t skiptr,skipte,skipteme,skiptreedraws;
   if(nkeeptrain) {skiptr=nd/nkeeptrain;}
   else skiptr = nd+1;
   if(nkeeptest) {skipte=nd/nkeeptest;}
   else skipte=nd+1;
   if(nkeeptestme) {skipteme=nd/nkeeptestme;}
   else skipteme=nd+1;
   if(nkeeptreedraws) {skiptreedraws = nd/nkeeptreedraws;}
   else skiptreedraws=nd+1;

   //--------------------------------------------------
   //print args
   printf("*****Data:\n");
   printf("data:n,p,np: %zu, %zu, %zu\n",n,p,np);
   printf("y1,yn: %lf, %lf\n",iy[0],iy[n-1]);
   printf("x1,x[n*p]: %lf, %lf\n",ix[0],ix[n*p-1]);
   if(np) printf("xp1,xp[np*p]: %lf, %lf\n",ixp[0],ixp[np*p-1]);
   printf("*****Number of Trees: %zu\n",m);
   printf("*****Number of Cut Points: %d ... %d\n", numcut[0], numcut[p-1]);
   printf("*****burn and ndpost: %zu, %zu\n",burn,nd);
   printf("*****Prior:beta,alpha,tau,nu,lambda: %lf,%lf,%lf,%lf,%lf\n",
                   mybeta,alpha,tau,nu,lambda);
   printf("*****sigma: %lf\n",sigma);
   printf("*****w (weights): %lf ... %lf\n",iw[0],iw[n-1]);
   cout << "*****Dirichlet:sparse,theta,omega,a,b,rho,augment: "
	<< dart << ',' << theta << ',' << omega << ',' << a << ','
	<< b << ',' << rho << ',' << aug << endl;
   printf("*****nkeeptrain,nkeeptest,nkeeptestme,nkeeptreedraws: %zu,%zu,%zu,%zu\n",
               nkeeptrain,nkeeptest,nkeeptestme,nkeeptreedraws);
   printf("*****printevery: %zu\n",printevery);
   printf("*****skiptr,skipte,skipteme,skiptreedraws: %zu,%zu,%zu,%zu\n",skiptr,skipte,skipteme,skiptreedraws);

   //--------------------------------------------------
   //LRF: add space for asymmetric error, u
   double* u_asym = new double[n];
   for(size_t i=0;i<n;i++) u_asym[i]=sigma;

   //LRF: add space for asymmetric error, u
   double* gamma_asym = new double[n];
   for(size_t i=0;i<n;i++) gamma_asym[i]=sigma;

   //LRF: add space for asymmetric error, u
   double* delta_asym = new double[n];
   for(size_t i=0;i<n;i++) delta_asym[i]=1;

   //LRF: add space for lambda, the location parameter of exponential
   double* u_lambda = new double[up];
   for(size_t i=0;i<up;i++) u_lambda[i]=lambda_start[i];
   //temporary storage for proposal of proposals
   double gamma_prop;
   double u_lambda_prop;
   double lambda_sd;
   //for(size_t i=0;i<up;i++) u_lambda_prop[i]=0;
   //computing acceptance probabilities
   double ll_prop;
   double ll_old;

   double scale_prop;
   double scale_old;

   double sig_prop;

   double rho_asym = 0.8;
   //--------------------------------------------------
   //heterbart bm(m);
   bm.setprior(alpha,mybeta,tau);
   bm.setdata(p,n,ix,iy,numcut);
   bm.setdart(a,b,rho,aug,dart,theta,omega);

   //--------------------------------------------------
   //sigma
   //gen.set_df(n+nu);
   double *svec = new double[n];
   for(size_t i=0;i<n;i++) svec[i]=iw[i]*sigma;

   //--------------------------------------------------

   std::stringstream treess;  //string stream to write trees to
   treess.precision(10);
   treess << nkeeptreedraws << " " << m << " " << p << endl;
   // dart iterations
   std::vector<double> ivarprb (p,0.);
   std::vector<size_t> ivarcnt (p,0);

   //--------------------------------------------------
   //temporary storage
   //out of sample fit
   double* fhattest=0; //posterior mean for prediction
   if(np) { fhattest = new double[np]; }
   double* captest=0; //posterior mean for prediction
   if(np) { captest = new double[np]; }

   double restemp=0.0,rss=0.0;


   //--------------------------------------------------
   //mcmc
   printf("\nMCMC\n");
   //size_t index;
   size_t trcnt=0; //count kept train draws
   size_t tecnt=0; //count kept test draws
   size_t temecnt=0; //count test draws into posterior mean
   size_t treedrawscnt=0; //count kept bart draws
   bool keeptest,keeptestme,keeptreedraw;

   time_t tp;
   int time1 = time(&tp);
   xinfo& xi = bm.getxinfo();

   for(size_t i=0;i<(nd+burn);i++) {
      if(i%printevery==0) printf("done %zu (out of %lu)\n",i,nd+burn);
      if(i==(burn/2)&&dart) bm.startdart();
      //---draw BETA (u_lambda)-----------------------------------------------

      for (size_t j=0; j < up; j++){
      ll_prop = 0;
      ll_old = 0;

      u_lambda_prop = u_lambda[j] + lambda_prop_sd[j]*gen.normal();

        for (size_t k=0;k<n;k++) {

          scale_prop = 0;
          scale_old = 0;
          for (size_t j2=0;j2<up;j2++){
            if (j2 == j){
              scale_prop += iux(k,j2)*u_lambda_prop ;
              scale_old  += iux(k,j2)*u_lambda[j2];
            }else{
            scale_prop += iux(k,j2)*u_lambda[j2] ;
            scale_old  += iux(k,j2)*u_lambda[j2];
            }
          }
        ll_prop += -scale_prop - gamma_asym[k]*exp(-scale_prop);
        //old log likelihood * proposal prior f(y[k] \mid u^(i-1), ..)
        ll_old  += -scale_old - gamma_asym[k]*exp(-scale_old);
        //ratio (diff) proposal / old compared to uniform
        }
        ll_prop +=-pow(2*2.5*2.5,-1)*pow((u_lambda_prop - 0),2); //N(0,1) prior on betas
        ll_old  +=-pow(2*2.5*2.5, -1)*pow((u_lambda[j] - 0),2); //N(0,1) prior on betas

      if (ll_prop - ll_old > log(gen.uniform())){
        u_lambda[j] = u_lambda_prop;
      }
      }

      //---draw gamma[1:n]-----------------------------------------------
      // RW Gibbs step to update u[1:n]
      for(size_t k=0;k<n;k++) {
        gamma_prop = gamma_asym[k] + gamma_prop_sd*gen.normal();
        if (gamma_prop > 0){
          scale_prop=0;
          for (size_t j2=0;j2<up;j2++){
              scale_prop +=iux(k,j2)*u_lambda[j2] ;
          }
        //DISTRIBUTION IS EXPONENTIAL (f(u) \propto exp(-u*XBETA))
        //proposal log likelihood * proposal prior f(y[k] \mid u_prop, ..)
        ll_prop = (-1/(2*sigma*sigma))*pow((iy[k] - (bm.f(k) - delta_asym[k]*gamma_prop)),2) - gamma_prop*exp(-scale_prop);
        //old log likelihood * proposal prior f(y[k] \mid u^(i-1), ..)
        ll_old = (-1/(2*sigma*sigma))*pow((iy[k] - (bm.f(k) - delta_asym[k]*gamma_asym[k])),2)-gamma_asym[k]*exp(-scale_prop);
        //ratio (diff) proposal / old compared to uniform

        if ( ll_prop - ll_old > log(gen.uniform())){
          gamma_asym[k] = gamma_prop;
        }// else don't update if the proposal is negative
        }
        u_asym[k] = gamma_asym[k]*delta_asym[k];
      }
      //---draw delta-----------------------------------------------
      double L0 = 0;
      double L1 = 0;
      for(size_t k=0;k<n;k++) {
      L1 = rho_asym*exp(-(1/(2*sigma*sigma))*pow(iy[k] - bm.f(k) + 1*gamma_asym[k],2));
      L0 = (1-rho_asym)*exp(-(1/(2*sigma*sigma))*pow(iy[k] - bm.f(k),2));
      delta_asym[k] = R::rbinom(1, L1/(L0 + L1) );

      }
      //---draw rho-----------------------------------------------
      double sumdelta = 0;

      for(size_t k=0;k<n;k++) {sumdelta += delta_asym[k];}
      // prior currently a = 2, b = 2
      rho_asym = R::rbeta(2 + sumdelta, 2 + n-sumdelta);

      rhodraw[i]=rho_asym;

      // update u
      for(size_t k=0;k<n;k++) {
        u_asym[k] = delta_asym[k]*gamma_asym[k];
        }

      //---draw bart-----------------------------------------------
      bm.draw(svec, gen, u_asym); // LRF added u

      //---draw sigma-----------------------------------------------
      rss=0.0;
      for(size_t k=0;k<n;k++) {restemp=(iy[k]-bm.f(k) + u_asym[k])/(iw[k]); rss += restemp*restemp;}

      //end LRF new
      sigma = sqrt((nu*lambda + rss)/gen.chi_square(n+nu));
      for(size_t k=0;k<n;k++) svec[k]=iw[k]*sigma;
      sdraw[i]=sigma;

      if(i>=burn) {
        // LRF: will later divide trmean by nd
         for(size_t k=0;k<n;k++) trmean[k]+=bm.f(k);
         if(nkeeptrain && (((i-burn+1) % skiptr) ==0)) {
            //index = trcnt*n;;
            //for(size_t k=0;k<n;k++) trdraw[index+k]=bm.f(k);
            for(size_t k=0;k<n;k++) {
              TRDRAW(trcnt,k)=bm.f(k);
              CAPDRAW(trcnt,k)=bm.f(k) + sigma*gen.normal() - u_asym[k];
              U_ASYM_DRAW(trcnt,k)=u_asym[k];
            }
            for(size_t k=0;k<up;k++){
              U_LAMBDA_DRAW(trcnt,k)=u_lambda[k];
            }
            trcnt+=1;
         }
         keeptest = nkeeptest && (((i-burn+1) % skipte) ==0) && np;
         keeptestme = nkeeptestme && (((i-burn+1) % skipteme) ==0) && np;
         if(keeptest || keeptestme) bm.predict(p,np,ixp,fhattest);
         if(keeptest) {
            //index=tecnt*np;
            //for(size_t k=0;k<np;k++) tedraw[index+k]=fhattest[k];
            for(size_t k=0;k<np;k++) TEDRAW(tecnt,k)=fhattest[k];
            for(size_t k=0;k<np;k++) { //over testing observations
              scale_prop=0;
              for (size_t j2=0;j2<up;j2++){
                scale_prop +=iuxp(k,j2)*u_lambda[j2] ;
              }
              TECAPDRAW(tecnt,k)=fhattest[k] + sigma*gen.normal()  - R::rbinom(1, rho_asym)*gen.gamma(1,exp(-scale_prop)) ;
            }
            tecnt+=1;
         }
         if(keeptestme) {
            for(size_t k=0;k<np;k++) temean[k]+=fhattest[k];
            temecnt+=1;
         }
         keeptreedraw = nkeeptreedraws && (((i-burn+1) % skiptreedraws) ==0);
         if(keeptreedraw) {
	   //#ifndef NoRcpp
//	   Rcpp::List lists(m*treesaslists);
//	   #endif

            for(size_t j=0;j<m;j++) {
	      treess << bm.gettree(j);
/*
	      #ifndef NoRcpp
	      varcount.row(treedrawscnt)=varcount.row(treedrawscnt)+bm.gettree(j).tree2count(p);
	      if(treesaslists) lists(j)=bm.gettree(j).tree2list(xi, 0., 1.);
	      #endif
*/
	    }
            #ifndef NoRcpp
//	    if(treesaslists) list_of_lists(treedrawscnt)=lists;
	    ivarcnt=bm.getnv();
	    ivarprb=bm.getpv();
	    size_t k=(i-burn)/skiptreedraws;
	    for(size_t j=0;j<p;j++){
	      varcnt(k,j)=ivarcnt[j];
	      //varcnt(i-burn,j)=ivarcnt[j];
	      varprb(k,j)=ivarprb[j];
	      //varprb(i-burn,j)=ivarprb[j];
	    }
            #else
	    varcnt.push_back(bm.getnv());
	    varprb.push_back(bm.getpv());
	    #endif

            treedrawscnt +=1;
         }
      }
   }
   int time2 = time(&tp);
   printf("time: %ds\n",time2-time1);
   for(size_t k=0;k<n;k++) trmean[k]/=nd;
   for(size_t k=0;k<np;k++) temean[k]/=temecnt;
   printf("check counts\n");
   printf("trcnt,tecnt,temecnt,treedrawscnt: %zu,%zu,%zu,%zu\n",trcnt,tecnt,temecnt,treedrawscnt);
   //--------------------------------------------------
   //PutRNGstate();

   if(fhattest) delete[] fhattest;
   if(svec) delete [] svec;

   //--------------------------------------------------
   //return
#ifndef NoRcpp
   Rcpp::List ret;
   ret["sigma"]=sdraw;
   ret["rho"]=rhodraw;
   ret["lambda"]=u_lambda_draw;
   //ret["u_asym"]=u_asym_draw;
   ret["potential.train.mean"]=trmean;
   ret["potential.train"]=trdraw;//=bm.f(k)
   ret["cap.train"] = capdraw;//CAPDRAW(trcnt,k)=bm.f(k) + sigma*gen.normal() - u_asym[k];
   ret["cap.test"]=tecapdraw; //TECAPDRAW(tecnt,k)=fhattest[k] + sigma*gen.normal()  - R::rbinom(1, rho_asym)*gen.gamma(1,exp(-scale_prop)) ;
   ret["u.train"] = u_asym_draw;
   ret["u_beta.train"] = u_lambda_draw;
   ret["potential.test.mean"]=temean;
   ret["potential.test"]=tedraw;//TEDRAW(tecnt,k)=fhattest[k];
   //ret["varcount"]=varcount;
   ret["varcount"]=varcnt;
   ret["varprob"]=varprb;

   //for(size_t i=0;i<m;i++) {
    //  bm.gettree(i).pr();
   //}

   Rcpp::List xiret(xi.size());
   for(size_t i=0;i<xi.size();i++) {
      Rcpp::NumericVector vtemp(xi[i].size());
      std::copy(xi[i].begin(),xi[i].end(),vtemp.begin());
      xiret[i] = Rcpp::NumericVector(vtemp);
   }

   Rcpp::List treesL;
   //treesL["nkeeptreedraws"] = Rcpp::wrap<int>(nkeeptreedraws); //in trees
   //treesL["ntree"] = Rcpp::wrap<int>(m); //in trees
   //treesL["numx"] = Rcpp::wrap<int>(p); //in cutpoints
   treesL["cutpoints"] = xiret;
   treesL["trees"]=Rcpp::CharacterVector(treess.str());
//   if(treesaslists) treesL["lists"]=list_of_lists;
   ret["treedraws"] = treesL;

   return ret;
#else

#endif

}
