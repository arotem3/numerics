#include <numerics.hpp>

double numerics::ode::rk4::_step(double k, double& t1, arma::vec& u1, arma::vec& f1, const odefunc& f, const odejacobian* jacobian) {
    const arma::vec& u = _prev_u.back();
    const double& t = _prev_t.back();

    arma::vec v = k*_prev_f.back();
    arma::vec p = u + rk4b[0]*v;
    for (short i=1; i < 5; ++i) {
        v = rk4a[i]*v + k*f(t+rk4c[i]*k, p);
        p += rk4b[i]*v;
    }
    
    t1 = t + k;
    u1 = std::move(p);
    f1 = f(t1,u1);
    return _cstep;
}