#include <numerics.hpp>

void numerics::optimization::LmLSQR::_initialize(const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) {
    _F = f(x);
    if (jacobian == nullptr) _J = approx_jacobian(f,x);
    else _J = (*jacobian)(x);

    _lam = _damping_param * arma::norm(_J.diag(), "inf");
}

bool numerics::optimization::LmLSQR::_step(arma::vec& dx, arma::vec& F1, const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) {
    arma::vec D; arma::mat U,V;
    bool success = arma::svd(U, D, V, _J);
    if (not success) {
        if (jacobian == nullptr) _J = approx_jacobian(f, x);
        else _J = (*jacobian)(x);
        success = arma::svd(U, D, V, _J);
        if (not success) return false;
    }
    double rho;
    double f0 = std::pow(arma::norm(_F),2);
    arma::vec JF = -_J.t() * _F;
    arma::vec UF = -U.t() * _F;
    while (true) {
        dx = V * (D/(arma::square(D)+_lam) % UF);
        F1 = f(x + dx);
        if (F1.has_nan()) return false;
        double f1 = std::pow(arma::norm(F1),2);
        rho = (f0-f1) / arma::dot(dx, _lam*dx + JF);
        if (rho > 0) {
            if (jacobian == nullptr) _J = approx_jacobian(f, x+dx);
            else _J = (*jacobian)(x+dx);

            _lam *= std::max(0.33, 1 - std::pow(2*rho-1,3));

            break;
        } else {
            _lam *= _damping_scale;
        }
    }
    return true;
}