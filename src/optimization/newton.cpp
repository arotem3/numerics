#include <numerics.hpp>

void numerics::optimization::Newton::_initialize(const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) {
    _F = f(x);
}

bool numerics::optimization::Newton::_step(arma::vec& dx, arma::vec& F1, const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) {
    bool success;
    if (jacobian == nullptr) {
        double h = 1e-6*std::max(1.0, arma::norm(x));
        VecFunc JacMult = [this, h, &F1, &f, &x](const arma::vec& v) -> arma::vec {
            double C = std::max(1.0, arma::norm(v)) / h;
            return C * (f(x + v/C) - _F);
        };
        success = numerics::optimization::gmres(dx, JacMult, -_F, _xtol);
    } else {
        _J = (*jacobian)(x);
        if (_J.has_nan()) return false;
        success = arma::solve(dx, _J, -_F);
    }
    if (not success) return false;
    auto line_f = [&F1,&x,&f,&dx](double a) -> double {
        F1 = f(x + a*dx);
        return arma::norm(F1);
    };
    if (line_f(1.0) > 0.99*arma::norm(_F)) {
        double a = numerics::optimization::fminbnd(line_f, 0.0, 1.0, 1e-2);
        dx *= a;
    }
    return true;
}

bool numerics::optimization::NewtonMin::_step(arma::vec& dx, arma::vec& g1, const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) {
    bool success;
    if (hessian == nullptr) {
        double h = 1e-6*std::max(1.0, arma::norm(x));
        VecFunc HessMult = [this,&df,&x,h](const arma::vec& v) -> arma::vec {
            double C = std::max(1.0, arma::norm(v)) / h;
            return C * (df(x + v/C) - _g);
        };
        success = pcg(dx, HessMult, -_g, _xtol);
        if (not success) success = gmres(dx, HessMult, -_g, _xtol);
    } else {
        arma::mat H = (*hessian)(x);
        success = arma::solve(dx, H, -_g);
    }

    if (not success) return false;

    double alpha = wolfe_step(f, df, x, dx);

    dx *= alpha;
    g1 = df(x+dx);

    return true;
}