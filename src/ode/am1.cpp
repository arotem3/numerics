#include <numerics.hpp>

void numerics::ode::am1::solve_ivp(const odefunc& f, double t0, double tf, const arma::vec& U0) {
    _check_range(t0,tf);
    _check_step(t0,tf);
    double k = _step;

    _t.clear(); _U.clear();
    _t.push_back(t0);
    _U.push_back(U0);

    optimization::Broyd fsolver(_max_solver_err, _max_solver_iter);

    unsigned long long i = 1;
    while (_t.back() < tf) {
        double tt = _next_t(k, _t.at(i-1), tf);

        arma::vec UU = _U.at(i-1);
        auto eulerStep = [this,&f,tt,i,k](const arma::vec& x) -> arma::vec {
            arma::vec z = x - (  _U.at(i-1) + k * f(tt, x)  );
            return z;
        };
        fsolver.fsolve(UU, eulerStep);

        double kk = event_handle(_t.at(i-1), _U.at(i-1), tt, UU, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }
        _t.push_back(tt);
        _U.push_back(std::move(UU));
        i++;
        if (kk == 0) break; // event stop
    }
}

void  numerics::ode::am1::solve_ivp(const odefunc& f, const odejacobian& jacobian, double t0, double tf, const arma::vec& U0) {
    _check_range(t0,tf);
    _check_step(t0,tf);
    double k = _step;

    _t.clear(); _U.clear();
    _t.push_back(t0);
    _U.push_back(U0);

    optimization::Newton fsolver(_max_solver_err, _max_solver_iter);

    unsigned long long i = 1;
    while (_t.back() < tf) {
        double tt = _next_t(k, _t.at(i-1), tf);

        arma::vec UU = _U.at(i-1);
        auto eulerStep = [this,&f,tt,i,k](const arma::vec& x) -> arma::vec {
            arma::vec z = x - (  _U.at(i-1) + k * f(tt, x)  );
            return z;
        };
        auto euler_jac = [&jacobian,tt,k](const arma::vec& x) -> arma::mat {
            arma::mat J = jacobian(tt, x);
            J = arma::eye(arma::size(J)) - k*J;
            return J;
        };
        fsolver.fsolve(UU, eulerStep, euler_jac);

        double kk = event_handle(_t.at(i-1), _U.at(i-1), tt, UU, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        _t.push_back(tt);
        _U.push_back(std::move(UU));
        i++;
        if (kk == 0) break; // event stop
    }
}