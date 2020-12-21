#include <numerics.hpp>

double numerics::ode::am2::_step(double k, double& t1, arma::vec& u1, arma::vec& f1, const odefunc& f, const odejacobian* jacobian) {
    arma::vec z = _prev_u.back() + 0.5*k*_prev_f.back();

    double t = _prev_t.back();

    auto am2f = [t,k,&z,&f,&f1](const arma::vec& v) -> arma::vec {
        f1 = f(t+k,v);
        return v - (z + 0.5*k*f1);
    };

    u1 = z;
    if (jacobian == nullptr) solver->fsolve(u1, am2f);
    else {
        auto am2J = [t,k,&z,&jacobian](const arma::vec& v) -> arma::mat {
            arma::mat J = (*jacobian)(t+k, v);
            return arma::eye(arma::size(J)) - 0.5*k*J;
        };
        solver->fsolve(u1, am2f, am2J);
    }
    t1 = t + k;
    return _cstep;
}