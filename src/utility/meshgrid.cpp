#include "numerics.hpp"

/* MESHGRID : produces a meshgrid from a single input vector,
 * the Y matrix complement to X is X.t().  */
arma::mat numerics::meshgrid(const arma::vec& x) {
    int n = x.n_elem;
    return arma::repmat(x.t(),n,1);
}