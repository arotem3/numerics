#include "numerics.hpp"

//--- produces a meshgrid from a single input vector ---//
//--- complementary vector to output is the transpose --//
arma::mat numerics::meshgrid(const arma::vec& x) {
    int n = x.n_elem;
    return arma::repmat(x.t(),n,1);
}