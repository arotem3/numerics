#include <numerics.hpp>

void numerics::optimization::LmLSQR::_initialize(const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) {
    _F = f(x);
    if (jacobian == nullptr) _J = approx_jacobian(f,x);
    else _J = (*jacobian)(x);

    _delta = 0.9*std::max(1.0, arma::norm(x,"inf"));
}

bool numerics::optimization::LmLSQR::_step(arma::vec& dx, arma::vec& F1, const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) {
    arma::vec D; arma::mat U,V;
    bool success = arma::svd_econ(U, D, V, _J);
    if (not success) {
        if (jacobian == nullptr) _J = approx_jacobian(f, x);
        else _J = (*jacobian)(x);
        success = arma::svd_econ(U, D, V, _J);
        if (not success) return false;
    }
    double rho;
    double f0 = 0.5*std::pow(arma::norm(_F),2);
    arma::vec JF = -_J.t() * _F;
    arma::vec UF = -U.t() * _F;

    arma::vec p = V * (UF / D);
    int nit = 0;
    double lam;
    while (true) {
        if (nit > 100) return false;

        if (arma::norm(p) <= _delta) {
            dx = p;
            lam = 0;
        } else {
            auto lam_search = [&](double l) -> double {
                dx = V * (D / (arma::square(D) + l*l) % UF);
                return 1/_delta - 1/arma::norm(dx);
            };
            lam = std::min(D.min()+1e-10, 1.0);
            lam = newton_1d(lam_search, lam, 1e-4);
            lam = lam * lam;
            dx = V * (D / (arma::square(D)+lam) % UF);
        }

        F1 = f(x + dx);
        if (F1.has_nan()) return false;

        arma::vec Jp = _J*dx;
        double f1 = 0.5*std::pow(arma::norm(F1),2);
        rho = (f1 - f0) / (arma::dot(_F + 0.5*Jp, Jp));
        
        if (rho < 0.25) _delta = 0.25*arma::norm(dx);
        else {
            if ((rho > 0.75) or (lam == 0)) _delta = std::min(2*_delta, 10.0);
        }
        
        if ((f1 < 0.99*f0) or (rho > 0.1)) {
            if (jacobian == nullptr) _J = approx_jacobian(f, x+dx);
            else _J = (*jacobian)(x + dx);
            break;
        }
        nit++;
    }
    
    return true;
}