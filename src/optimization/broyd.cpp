#include <numerics.hpp>

void numerics::optimization::Broyden::_initialize(const arma::vec& x, const VecFunc& f, const MatFunc* jacobian)  {
    _F = f(x);
    if (jacobian == nullptr) _J = numerics::approx_jacobian(f,x);
    else _J = (*jacobian)(x);

    _J = arma::pinv(_J);
}

bool numerics::optimization::Broyden::_step(arma::vec& dx, arma::vec& F1, const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) {
    if (_J.has_nan()) return false;

    // bool success = arma::solve(dx, _J, -_F);
    dx = -_J * _F;
    // if (not success) {
    //     if (jacobian == nullptr) _J = approx_jacobian(f, x);
    //     else _J = (*jacobian)(x);

    //     success = arma::solve(dx, _J, -_F);
    //     if (not success) return false;
    // }
    if (dx.has_nan()) {
        if (jacobian == nullptr) _J = numerics::approx_jacobian(f, x);
        else _J = (*jacobian)(x);

        _J = arma::pinv(_J);
        dx = -_J * _F;
        if (dx.has_nan()) return false;
    }


    auto line_f = [&F1,&x,&f,&dx](double a) -> double {
        F1 = f(x + a*dx);
        return arma::norm(F1);
    };
    if (line_f(1.0) > 0.99*arma::norm(_F)) {
        double a = fminbnd(line_f, 0.0, 1.0, 1e-2);
        dx *= a;
    }

    arma::vec y = F1 - _F;
    // _J += (y - _J*dx)*dx.t() / arma::dot(dx, dx);
    arma::vec Jy = _J * y;
    _J += (dx - Jy) * dx.t() * _J / arma::dot(dx, Jy);

    return true;
}