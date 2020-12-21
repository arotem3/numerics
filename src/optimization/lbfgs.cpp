#include <numerics.hpp>

void numerics::optimization::LBFGS::_lbfgs_update(arma::vec& p) {
    long k = _S.size();
    if (k > 0) {
        arma::vec ro = arma::zeros(k);
        for (long i=0; i < k; ++i) {
            ro(i) = 1 / arma::dot(_S.at(i),_Y.at(i));
        }

        arma::vec q = p;
        arma::vec alpha = arma::zeros(k);
        
        for (long i(k-1); i >= 0; --i) {
            alpha(i) = ro(i) * arma::dot(_S.at(i),q);
            q -= alpha(i) * _Y.at(i);
        }

        arma::vec r = q * _hdiag;

        for (long i(0); i < k; ++i) {
            double beta = ro(i) * arma::dot(_Y.at(i),r);
            r += _S.at(i) * (alpha(i) - beta);
        }

        p = r;
    }
}

void numerics::optimization::LBFGS::_initialize(const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) {
    _g = df(x);
    _S.clear();
    _Y.clear();
}

bool numerics::optimization::LBFGS::_step(arma::vec& dx, arma::vec& g1, const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) {
    arma::vec p = -_g;
    _lbfgs_update(p);

    double alpha = wolfe_step(f, df, x, p, _wolfe_c1, _wolfe_c2);

    dx = alpha*p;

    if (dx.has_nan()) return false;

    g1 = df(x + dx);
    arma::vec y = g1 - _g;

    _hdiag = arma::dot(dx, y) / arma::dot(y,y);
    _S.push(dx);
    _Y.push(std::move(y));

    return true;
}