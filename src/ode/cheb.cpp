#include <numerics.hpp>

/* cheb(D, x, L, R, m) : constructs the Chebyshev spectral differentiation matrix.
 * --- D : cheb matrix storage.
 * --- x : grid value storage.
 * --- L,R : limits on x.
 * --- m : number of points. */
void numerics::ode::cheb(arma::mat& D, arma::vec& x, double L, double R, uint m) {
    m = m-1;
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

/* cheb(D, x, m) : constructs the Chebyshev matrix on the unit interval [-1,1].
 * --- D : cheb matrix storage.
 * --- x : grid storage.
 * --- m : number of points on interval. */
void numerics::ode::cheb(arma::mat& D, arma::vec& x, uint m) {
    cheb(D,x,-1,1,m);
}