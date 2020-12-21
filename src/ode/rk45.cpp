#include <numerics.hpp>
void numerics::ode::rk45::_next_step_size(double& k, double res, double tol) {
    k *= std::min(10.0, std::max(0.1, 0.9*std::pow(tol/res,0.2)));
    if (k < _min_step) throw std::runtime_error("rk45 could not converge b/c current step-size (=" + std::to_string(k) + ") < minimum step size (=" + std::to_string(_min_step) + ").");
}

double numerics::ode::rk45::_step(double k, double& t1, arma::vec& u1, arma::vec& f1, const odefunc& f, const odejacobian* jacobian) {
    arma::vec V1, V2, V3, V4, V5, V6, rk4;
    double res = 0;
    
    const arma::vec& u = _prev_u.back();
    const double& t = _prev_t.back();
    
    double tol = std::max(_atol, _rtol*arma::norm(u,"inf"));
    
    do {
        V1 = k * _prev_f.back();
        V2 = k * f( t + 0.2*k, u + 0.2*V1 );
        V3 = k * f( t + 0.3*k, u + (3.0/40)*V1 + (9.0/40)*V2 );
        V4 = k * f( t + 0.8*k, u + (44.0/45)*V1 - (56.0/15)*V2 + (32.0/9)*V3 );
        V5 = k * f( t + (8.0/9)*k, u + (19372.0/6561)*V1 - (25360.0/2187)*V2 + (64448.0/6561)*V3 - (212.0/729)*V4 );
        V6 = k * f( t + k, u + (9017.0/3168)*V1 - (355.0/33)*V2 + (46732.0/5247)*V3 + (49.0/176)*V4 - (5103.0/18656)*V5 );
        rk4 = u + (35.0/384)*V1 + (500.0/1113)*V3 + (125.0/192)*V4 - (2187.0/6784)*V5 + (11.0/84)*V6;
        u1 = u + (5179.0/57600)*V1 + (7571.0/16695)*V3 + (393.0/640)*V4 - (92097.0/339200)*V5 + (187.0/2100)*V6 + (1.0/40)*k*f(t + k, rk4);
        res = arma::norm(u1 - rk4, "inf");
        t1 = t + k;
        _next_step_size(k, res, tol);
    } while (res > tol);

    f1 = f(t1, u1);
    return k;
}
