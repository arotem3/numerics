#include "numerics.hpp"

/* polyder(p, k) : return the k^th derivative of a polynomial.
 * --- p : polynomial to differentiate.
 * --- k : the derivative order (k = 1 by default, i.e. first derivative). */
arma::vec numerics::polyder(const arma::vec& p, uint k) {
    if (p.n_elem <= 1) return arma::zeros(1);
    int n = p.n_elem - 1;
    arma::vec dp = arma::zeros(n);
    for (uint i=0; i < n; ++i) {
        dp(i) = (n-i) * p(i);
    }
    if (k > 1) return polyder(dp, k-1);
    else return dp;
}

/* polyint(p, c) : return the integral of a polynomial.
 * --- p : polynomial to integrate.
 * --- c : integration constant. */
arma::vec numerics::polyint(const arma::vec& p, double c) {
    int n = p.n_elem + 1;
    arma::vec ip = arma::zeros(n);
    for (uint i=0; i < n-1; ++i) {
        ip(i) = p(i) / (n-1-i);
    }
    ip(n-1) = c;
    return ip;
}