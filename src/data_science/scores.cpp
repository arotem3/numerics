#include <numerics.hpp>

double numerics::mse_score(const arma::vec& y, const arma::vec& yhat) {
    return arma::mean(arma::square(arma::conv_to<arma::vec>::from(y - yhat)));
}

double numerics::r2_score(const arma::vec& y, const arma::vec& yhat) {
    return 1.0 - mse_score(y, yhat)/arma::var(arma::conv_to<arma::vec>::from(y),1);
}

double numerics::accuracy_score(const arma::uvec& y, const arma::uvec& yhat) {
    return (long double)arma::sum(y == yhat) / y.n_elem;
}