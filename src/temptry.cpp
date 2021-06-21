#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double norm(double x, double y) {
  return sqrt(x*x + y*y);
}