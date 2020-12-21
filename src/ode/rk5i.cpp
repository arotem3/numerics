#include <numerics.hpp>

arma::vec numerics::ode::rk5i::_solve_v2(double k, double t, const arma::vec& z, const odefunc& f, const odejacobian* jacobian) {
    arma::vec v2 = z;
    auto v2f = [t,k,&z,&f](const arma::vec& v) -> arma::vec {
        return v - f(t+0.25*k, z + 0.125*k*v);
    };
    if (jacobian == nullptr) solver->fsolve(v2, v2f);
    else {
        auto v2J = [t,k,&z,&jacobian](const arma::vec& v) -> arma::mat {
            arma::mat J = (*jacobian)(t+0.25*k, z + 0.125*k*v);
            J = arma::eye(arma::size(J)) - 0.125*k*J;
            return J;
        };
        solver->fsolve(v2, v2f, v2J);
    }
    return v2;
}

arma::vec numerics::ode::rk5i::_solve_v3(double k, double t, const arma::vec& z, const odefunc& f, const odejacobian* jacobian) {
    arma::vec v3 = z;
    auto v3f = [t,k,&f,&z](const arma::vec& v) -> arma::vec {
        return v - f(t + 0.7*k, z + 0.15*k*v);
    };
    if (jacobian == nullptr) solver->fsolve(v3, v3f);
    else {
        auto v3J = [t,k,&jacobian,&z](const arma::vec& v) -> arma::mat {
            arma::mat J = (*jacobian)(t + 0.7*k, z + 0.15*k*v);
            J = arma::eye(arma::size(J)) - 0.15*k*J;
            return J;
        };
        solver->fsolve(v3, v3f, v3J);
    }
    return v3;
}

double numerics::ode::rk5i::_step(double k, double& t1, arma::vec& u1, arma::vec& f1, const odefunc& f, const odejacobian* jacobian) {
    arma::vec v2, v3, v4, z;

    const arma::vec& v1 = _prev_f.back();
    const arma::vec& u = _prev_u.back();
    const double& t = _prev_t.back();

    z = u + 0.125*k*v1;
    v2 = _solve_v2(k,t,z,f,jacobian);

    z = u + k*(-0.01*v1 + 0.56*v2);
    v3 = _solve_v3(k,t,z,f,jacobian);
    
    v4 = f(t+k, u + k*((2.0/7)*v1 + (5.0/7)*v3));
    z = u + k * ( (1.0/14)*v1 + (32.0/81)*v2 + (250.0/567)*v3 + (5.0/54)*v4 );

    t1 = t + k;
    u1 = std::move(z);
    f1 = f(t1,u1);
    return _cstep;
}