#include "numerics.hpp"

/* MESHGRID : produces meshgrids from two different input vectors.
 * --- xgrid : matrix to assign x values mesh to.
 * --- ygrid : matrix to assign y values mesh to.
 * --- x : x values.
 * --- y : y values. */
void numerics::meshgrid(arma::mat& xgrid, arma::mat& ygrid, const arma::vec& x, const arma::vec& y) {
    xgrid = arma::repmat(x.t(), y.n_elem, 1);
    ygrid = arma::repmat(y, 1, x.n_elem);
}

/* MESHGRID : produces meshgrid from single input vector, correspondingly ygrid = xgrid.t()
 * --- xgrid : matrix to assign x values to.
 * --- x : x values. */
void numerics::meshgrid(arma::mat& xgrid, const arma::vec& x) {
    xgrid = arma::repmat(x.t(), x.n_elem, 1);
}