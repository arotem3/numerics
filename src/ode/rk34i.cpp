#include <numerics.hpp>

void numerics::ode::rk34i::_next_step_size(double& k, double res, double tol) {
    k *= std::min(10.0, std::max(0.1, 0.9*std::pow(tol/res,0.25)));
    if (k < _min_step) throw std::runtime_error("method could not converge b/c current step-size (=" + std::to_string(k) + ") < minimum step size (=" + std::to_string(_min_step) + ").");
}

arma::vec numerics::ode::rk34i::_v_solve(double tv, double k, const arma::vec& z, arma::vec& f1, const odefunc& f, const odejacobian* jacobian) {
    arma::vec vv = z;
    auto vf = [tv,k,&f,&z,&f1](const arma::vec& v) -> arma::vec {
        f1 = f(tv, z + v/4); // stores this value in f1 for reuse at next time-step
        return v - k*f1;
    };

    if (jacobian == nullptr) solver->fsolve(vv, vf);
    else {
        auto vJ = [tv,k,&f,&jacobian,&z](const arma::vec& v) -> arma::mat {
            arma::mat J = (*jacobian)(tv, z + v/4);
            return arma::eye(arma::size(J)) - k/4*J;
        };
        solver->fsolve(vv, vf, vJ);
    }
    return vv;
}

double numerics::ode::rk34i::_step(double k, double& t1, arma::vec& u1, arma::vec& f1, const odefunc& f, const odejacobian* jacobian) {
    arma::vec v1, v2, v3, v4, v5, z, rk3;
    double res = 0;
    const arma::vec& u = _prev_u.back();
    const double& t = _prev_t.back();

    double tol = std::max(_atol, _rtol*arma::norm(u,"inf"));

    do {
        v1 = _v_solve(t+0.25*k, k, u, f1, f, jacobian);

        z = u + v1/2;
        v2 = _v_solve(t+0.75*k, k, z, f1, f, jacobian);

        z = u + (17.0/50)*v1 - (1.0/25)*v2;
        v3 = _v_solve(t+(11.0/20)*k, k, z, f1, f, jacobian);

        z = u + (371.0/1360)*v1 - (137.0/2720)*v2 + (15.0/544)*v3;
        v4 = _v_solve(t+0.5*k, k, z, f1, f, jacobian);

        z = u + (25.0/24)*v1 - (49.0/48)*v2 + (125.0/16)*v3 - (85.0/12)*v4;
        v5 = _v_solve(t+k, k, z, f1, f, jacobian);

        rk3 = u + (59.0/48)*v1 - (17.0/96)*v2 + (225.0/32)*v3 - (85.0/12)*v4;
        u1 = z + v5/4;
        t1 = t + k;

        res = arma::norm(u1 - rk3, "inf");
        _next_step_size(k, res, tol);
    } while(res > tol);
    return k;
}