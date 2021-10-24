#include <numerics.hpp>

arma::mat numerics::cubic_kernel(const arma::mat& x) {
    arma::mat K;
    K.set_size(x.n_rows, x.n_rows);
    for (u_int i=0; i < x.n_rows; ++i) {
        for (u_int j=0; j < i; ++j) {
            K(i,j) = std::pow(arma::norm(x.row(i) - x.row(j)), 3);
            K(j,i) = K(i,j);
        }
        K(i,i) = 0;
    }
    return K;
}

arma::mat numerics::cubic_kernel(const arma::mat& x1, const arma::mat& x2) {
    if (x1.n_cols != x2.n_cols) {
        throw std::invalid_argument("require x1.n_cols (=" + std::to_string(x1.n_cols) + ") == x2.n_cols (=" + std::to_string(x2.n_cols) + ")");
    }
    arma::mat K;
    K.set_size(x2.n_rows, x1.n_rows);
    for (u_int i=0; i < x2.n_rows; ++i) {
        for (u_int j=0; j < x1.n_rows; ++j) {
            K(i,j) = std::pow(arma::norm(x1.row(j) - x2.row(i)), 3);
        }
    }
    return K;
}