#include <numerics.hpp>

bool numerics::optimization::Broyden::_step(arma::vec& dx, arma::vec& F1, const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) {
    if (_J.has_nan()) return false;

    bool success = arma::solve(dx, _J, -_F);
    if (not success) {
        if (jacobian == nullptr) _J = approx_jacobian(f, x);
        else _J = (*jacobian)(x);

        success = arma::solve(dx, _J, -_F);
        if (not success) return false;
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
    _J += (y - _J*dx)*dx.t() / arma::dot(dx, dx);

    return true;
}