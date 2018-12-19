#include "ODE.hpp"

void ODE::cheb(arma::mat& D, arma::vec& x, double L, double R, size_t m) {
    x = arma::regspace(0,m);
    x = -1*arma::cos(M_PI * x / m); // standard cheb nodes on [-1,1]
    x = (R - L) * x/2 + (L + R)/2; // transformation from [-1,1] -> [L,R]

    arma::vec c = arma::ones(m+1); c(0) = 2; c(m) = 2;
    c( arma::regspace<arma::uvec>(1,2,m) ) *= -1; // every other element negative

    D = arma::repmat(x,1,m+1);
    D -= D.t();
    D = c * ( 1/c.t() ) / (D + arma::eye(m+1, m+1));
    D -= arma::diagmat(arma::sum(D,1));
}