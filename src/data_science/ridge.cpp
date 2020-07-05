#include <numerics.hpp>

/* ridge_cv(use_cgd=false, tol=1e-5) : initializes ridge_cv object. */
numerics::ridge_cv::ridge_cv() : coef(_w), residuals(_res), cov_eigvecs(_eigvecs), cov_eigvals(_eigvals), regularizing_param(_lambda), RMSE(_rmse), eff_df(_df) {}

/* fit(X,y) : fit a ridge regression model using the generalized formula for LOOCV.
 * --- X : array of indpendent variable data, where each row is data point.
 * --- y : array of dependent variable data, where each row is a data point. */
void numerics::ridge_cv::fit(const arma::mat& X, const arma::mat& y) {
    uint n_obs = X.n_rows;
    arma::mat covariance = X.t() * X;
    arma::mat yp = X.t() * y;
    arma::eig_sym(_eigvals, _eigvecs, covariance);
    yp = _eigvecs.t() * yp;
    auto GCV = [&](double lam) -> double { //
        _w = _eigvecs * arma::diagmat(1 / (_eigvals + lam)) * yp;
        _df = arma::sum(_eigvals / (_eigvals + lam));
        _res = y - X*_w;
        _rmse = arma::norm(_res,"fro");
        return std::pow(_rmse / (n_obs-_df), 2) * n_obs;
    };
    _lambda = numerics::fminbnd(GCV, 0, 1e4);
    _rmse /= _res.n_elem;
}