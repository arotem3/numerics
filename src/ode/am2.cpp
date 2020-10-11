#include <numerics.hpp>

void numerics::ode::am2::solve_ivp(const odefunc& f, double t0, double tf, const arma::vec& U0) {
    _check_range(t0, tf);
    _check_step(t0, tf);
    double k = _step;

    _t.clear(); _U.clear();
    _t.push_back(t0);
    _U.push_back(U0);
    
    _t.push_back(t0 + k);

    optimization::Broyd fsolver(_max_solver_err, _max_solver_iter);

    arma::vec V = U0;
    auto backEulerStep = [this,&f,&U0,k](const arma::vec& x) -> arma::vec {
        arma::vec z = U0 + k * f(_t.at(1),x);
        return x - z;
    };
    fsolver.fsolve(V,backEulerStep);
    _U.push_back(std::move(V));

    unsigned long long i = 2;
    while (_t.back() < tf) {
        double tt = _next_t(k, _t.at(i-1), tf);

        V = _U.at(i-1);
        auto trapStep = [this,&f,tt,i,k](const arma::vec& x) -> arma::vec {
            arma::vec z = _U.at(i-1) + (k/2) * ( f(_t.at(i-1),_U.at(i-1)) + f(tt, x) );
            return x - z;
        };
        fsolver.fsolve(V,trapStep);

        double kk = event_handle(_t.at(i-1), _U.at(i-1), tt, V, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }
        _t.push_back(tt);
        _U.push_back(std::move(V));
        i++;
        if (kk == 0) break; // event stop
    }
}

void numerics::ode::am2::solve_ivp(const odefunc& f, const odejacobian& jacobian, double t0, double tf, const arma::vec& U0) {
    _check_range(t0, tf);
    _check_step(t0, tf);
    double k = step;

    _t.clear(); _U.clear();
    _t.push_back(t0);
    _U.push_back(U0);
    
    _t.push_back(t0 + k);

    optimization::Newton fsolver(_max_solver_err, _max_solver_iter);

    arma::vec V = U0;
    auto backEulerStep = [this,&f,&U0,k](const arma::vec& x) -> arma::vec {
        arma::vec z = U0 + k * f(_t.at(1),x);
        return x - z;
    };
    auto euler_jac = [this,&jacobian,&U0,k](const arma::vec& x) -> arma::mat {
        arma::mat J = jacobian(_t.at(1),x);
        J = arma::eye(arma::size(J)) - k*J;
        return J;
    };
    fsolver.fsolve(V, backEulerStep, euler_jac);
    _U.push_back(std::move(V));

    unsigned long long i = 2;
    while (_t.back() < tf) {
        double tt = _next_t(k, _t.at(i-1), tf);

        V = _U.at(i-1);
        auto trapStep = [this,&f,tt,i,k](const arma::vec& x) -> arma::vec {
            arma::vec z = _U.at(i-1) + (k/2) * ( f(_t.at(i-1),_U.at(i-1)) + f(tt, x) );
            return x - z;
        };
        auto trap_jac = [&jacobian,tt,i,k](const arma::vec& x) -> arma::mat {
            arma::mat J = jacobian(tt, x);
            J = arma::eye(arma::size(J)) - (k/2) * J;
            return J;
        };
        fsolver.fsolve(V,trapStep,trap_jac);

        double kk = event_handle(_t.at(i-1), _U.at(i-1), tt, V, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }
        _t.push_back(tt);
        _U.push_back(std::move(V));
        i++;
        if (kk == 0) break; // event stop
    }
}