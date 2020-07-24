#include <numerics.hpp>

void numerics::RidgeCV::fit(const arma::mat& X, const arma::vec& y) {
    _check_xy(X,y);
    _dim = X.n_cols;
    uint n_obs = X.n_rows;

    arma::mat covariance;
    arma::vec yp;
    arma::mat P;

    if (_fit_intercept) {
        P = _add_intercept(X);
        covariance = P.t() * P;
        yp = P.t() * y;
    } else {
        covariance = X.t() * X;
        yp = X.t() * y;
    }
    arma::eig_sym(_eigvals, _eigvecs, covariance);
    yp = _eigvecs.t() * yp;
    auto GCV = [&](double lam) -> double {
        lam = std::pow(10.0, lam);
        _w = _eigvecs * (yp / (_eigvals + lam));

        _df = arma::sum(_eigvals / (_eigvals + lam));

        double mse;
        if (_fit_intercept) mse = mse_score(y, P*_w);
        else mse = mse_score(y, X*_w);

        return mse * std::pow(n_obs / (n_obs-_df), 2);
    };
    _lambda = optimization::fminbnd(GCV, -8, 4);
    _lambda = std::pow(10.0, _lambda);
    _split_weights();
}

arma::vec numerics::RidgeCV::predict(const arma::mat& X) const {
    _check_x(X);
    return _b + X * _w;
}

double numerics::RidgeCV::score(const arma::mat& x, const arma::vec& y) const {
    _check_xy(x, y);
    return r2_score(y, predict(x));
}