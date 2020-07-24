#include "numerics.hpp"

void numerics::Splines::set_lambda(double l) {
    if (_fitted) throw std::runtime_error("cannot set lambda after call to fit.");
    if (_use_df) throw std::runtime_error("cannot set lambda after setting df.");
    if (_use_lambda) throw std::runtime_error("lambda already set.");
    if (l < 0) throw std::invalid_argument("require lambda (=" + std::to_string(l) + ") >= 0");

    _lambda = l;
    _use_lambda = true;
}

void numerics::Splines::set_df(double df) {
    if (_fitted) throw std::runtime_error("cannot set df after call to fit.");
    if (_use_df) throw std::runtime_error("df already set.");
    if (_use_lambda) throw std::runtime_error("cannot set df after setting lambda.");
    if (df < 1) throw std::invalid_argument("require df (=" + std::to_string(df) + ") >= 1.");

    _df = df;
    _use_df = true;
}

void numerics::Splines::fit(const arma::mat& x, const arma::vec& y) {
    if (_fitted) {
        throw std::runtime_error("Splines object already fitted.");
    }
    _check_xy(x,y);
    if (_use_df and (_df > x.n_rows)) {
        throw std::runtime_error("requested degrees of freedom (=" + std::to_string(_df) + ") exceeds the number of observations (=" + std::to_string(x.n_rows) + ").");
    }
    _dim = x.n_cols;
    _X = x;
    arma::mat P = _add_intercept(_X);
    _w = arma::solve(P, y);

    arma::mat K = cubic_kernel(_X);
    arma::eig_sym(_eigvals, _eigvecs, K);
    arma::vec D2 = arma::pow(_eigvals,2);

    arma::vec res = y - P*_w;
    arma::vec Vres = _eigvecs.t() * res;
    _split_weights();
    if (_use_df) { // df specified, infer lambda
        if (_df == _X.n_rows) _lambda = 0;
        else {
            auto g = [&](double L) -> double {
                return _df - arma::sum(D2 / (D2 + L));
            };
            double lower = D2.min(), upper = D2.max();
            _lambda = optimization::fzero(g, lower, upper, std::min(1e-8, lower/2));
        }
    } else if (not _use_lambda) { // neither df nor lambda specified, use generalized LOO cross-validation to find best lambda
        auto GCV = [&](double L) -> double {
            L = std::pow(10.0,L);
            _c = _eigvecs * ((_eigvals / (D2 + L)) % Vres);
            _df = arma::sum(D2 / (D2 + L));
            
            double mse = mse_score(res, K*_c);
            return mse * std::pow(_X.n_rows / (_X.n_rows - _df), 2);
        };
        _lambda = optimization::fminbnd(GCV, -8, 4);
        _lambda = std::pow(10.0, _lambda);
    }
    if (_use_lambda or _use_df) { // we have lambda, now compute _c
        _c = _eigvecs * arma::diagmat(_eigvals / (D2 + _lambda)) * Vres;
        if (_use_lambda) _df = arma::sum(D2 / (D2 + _lambda)); // infer df from lambda
    }
    _fitted = true;
}

arma::vec numerics::Splines::predict(const arma::mat& x) const {
    _check_x(x);
    if (not _fitted) {
        throw std::runtime_error("Splines object not fitted.");
    }
    return _b + x * _w + cubic_kernel(_X, x) * _c;
}

double numerics::Splines::score(const arma::mat& x, const arma::vec& y) const {
    _check_xy(x,y);
    return r2_score(y, predict(x));
}
