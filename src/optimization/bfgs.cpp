#include <numerics.hpp>

void numerics::optimization::BFGS::_initialize(const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) {
    _g = df(x);
    if (hessian == nullptr) {
        if (_use_fd) {
            _H = arma::symmatu(numerics::approx_jacobian(df,x));
            bool chol = arma::inv_sympd(_H, _H);
            if (!chol) _H = arma::pinv(_H);
        } else _H = arma::eye(x.n_elem, x.n_elem);
    } else {
        _H = (*hessian)(x);
        bool chol = arma::inv_sympd(_H, _H);
        if (!chol) _H = arma::pinv(_H);
    }
}

bool numerics::optimization::BFGS::_step(arma::vec& dx, arma::vec& g1, const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) {
    dx = -(_H*_g);

    if (dx.has_nan()) {
        if (hessian == nullptr) {
            if (_use_fd) {
                _H = arma::symmatu(numerics::approx_jacobian(df,x));
                bool chol = arma::inv_sympd(_H, _H);
                if (!chol) _H = arma::pinv(_H);
            } else _H = arma::eye(x.n_elem, x.n_elem);
        } else {
            _H = (*hessian)(x);
            bool chol = arma::inv_sympd(_H, _H);
            if (!chol) _H = arma::pinv(_H);
        }
        dx = -(_H*_g);
        if (dx.has_nan()) return false;
    }

    double alpha = wolfe_step(f, df, x, dx, _wolfe_c1, _wolfe_c2);

    dx *= alpha;
    g1 = df(x+dx);

    arma::vec y = g1 - _g;

    double sy = arma::dot(dx,y);
    arma::vec Hy = _H*y;
    double yHy = arma::dot(y,Hy);
    arma::mat sHy = dx*Hy.t() / sy;

    _H += (sy + yHy)/std::pow(sy,2) * dx*dx.t() - sHy - sHy.t();

    return true;
}
