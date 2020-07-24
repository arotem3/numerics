#include <numerics.hpp>

numerics::ode::ODESolution numerics::ode::am1::ode_solve(const odefunc& f, double t0, double tf, const arma::vec& U0) {
    _check_range(t0,tf);
    _check_step(t0,tf);
    double k = _step;

    u_long m = U0.n_elem;
    ODESolution sol(m);

    std::vector<double>& t = sol._tvec;
    std::vector<arma::vec>& U = sol._Uvec;
    t.push_back(t0);
    U.push_back(U0);

    optimization::Broyd fsolver(_max_solver_err, _max_solver_iter);

    unsigned long long i = 1;
    while (t.back() < tf) {
        double tt = _next_t(k, t.at(i-1), tf);

        arma::vec UU = U.at(i-1);
        auto eulerStep = [&f,&U,tt,i,k](const arma::vec& x) -> arma::vec {
            arma::vec z = x - (  U.at(i-1) + k * f(tt, x)  );
            return z;
        };
        fsolver.fsolve(UU, eulerStep);

        double kk = event_handle(t.at(i-1), U.at(i-1), tt, UU, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }
        t.push_back(tt);
        U.push_back(std::move(UU));
        i++;
        if (kk == 0) break; // event stop
    }
    sol._convert();
    return sol;
}

numerics::ode::ODESolution numerics::ode::am1::ode_solve(const odefunc& f, const odejacobian& jacobian, double t0, double tf, const arma::vec& U0) {
    _check_range(t0,tf);
    _check_step(t0,tf);
    double k = _step;

    u_long m = U0.n_elem;
    ODESolution sol(m);

    std::vector<double>& t = sol._tvec;
    std::vector<arma::vec>& U = sol._Uvec;
    t.push_back(t0);
    U.push_back(U0);

    optimization::Newton fsolver(_max_solver_err, _max_solver_iter);

    unsigned long long i = 1;
    while (t.back() < tf) {
        double tt = _next_t(k, t.at(i-1), tf);

        arma::vec UU = U.at(i-1);
        auto eulerStep = [&f,&U,tt,i,k](const arma::vec& x) -> arma::vec {
            arma::vec z = x - (  U.at(i-1) + k * f(tt, x)  );
            return z;
        };
        auto euler_jac = [&jacobian,tt,k](const arma::vec& x) -> arma::mat {
            arma::mat J = jacobian(tt, x);
            J = arma::eye(arma::size(J)) - k*J;
            return J;
        };
        fsolver.fsolve(UU, eulerStep, euler_jac);

        double kk = event_handle(t.at(i-1), U.at(i-1), tt, UU, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        t.push_back(tt);
        U.push_back(std::move(UU));
        i++;
        if (kk == 0) break; // event stop
    }
    sol._convert();
    return sol;
}