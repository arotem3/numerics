#include "numerics.hpp"

/* EVAL : evaluates f at every column of X. */
arma::vec numerics::eval(const vec_dfunc& f, arma::mat& X) {
    int n = X.n_cols;
    arma::vec F(n,arma::fill::zeros);
    for (int i(0); i < n; ++i) {
        F(i) = f(X.col(i));
    }
    return F;
}