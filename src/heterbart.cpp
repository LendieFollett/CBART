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

#include "heterbart.h"

//--------------------------------------------------
void heterbart::pr()
{
   cout << "+++++heterbart object:\n";
   bart::pr();
}
//--------------------------------------------------
void heterbart::draw(double *sigma, rn& gen, double *u_asym)// LRF added double *u
{
  //printf("*****MODIFIED IT heterbart::draw in src!!!!!! \n");
  //this is what is called by cwbart in bm.draw()
   for(size_t j=0;j<m;j++) {
      fit(t[j],xi,p,n,x,ftemp); //treefuns::fit
      for(size_t k=0;k<n;k++) {
         allfit[k] = allfit[k]-ftemp[k]; 
         r[k] = y[k]-allfit[k] + u_asym[k];// LRF: residual incorporates u
      }
      heterbd(t[j],xi,di,pi,sigma,nv,pv,aug,gen); //heterbd::heterbd p(T | r, sigma)
      heterdrmu(t[j],xi,di,pi,sigma,gen); //heterbartfuns::heterdrmu p(mu | T, sigma, r)
      fit(t[j],xi,p,n,x,ftemp);//treefuns::fit
      for(size_t k=0;k<n;k++) allfit[k] += ftemp[k];
   }
   //printf("LRF WHAT IS y1,yn: %lf, %lf\n",y[0],y[n-1]);
// y used in this function is simply y - mean(y)
   if(dartOn) {
     draw_s(nv,lpv,theta,gen);
     draw_theta0(const_theta,theta,lpv,a,b,rho,gen);
     for(size_t j=0;j<p;j++) pv[j]=::exp(lpv[j]);
   }
}
