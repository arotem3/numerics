#include <numerics.hpp>

numerics::ode::ODESolution numerics::ode::am2::ode_solve(const odefunc& f, double t0, double tf, const arma::vec& U0) {
    _check_range(t0, tf);
    _check_step(t0, tf);
    double k = _step;

    u_long m = U0.n_elem;
    ODESolution sol(m);

    std::vector<double>& t = sol._tvec;
    std::vector<arma::vec>& U = sol._Uvec;
    t.push_back(t0);
    U.push_back(U0);
    
    t.push_back(t0 + k);

    optimization::Broyd fsolver(_max_solver_err, _max_solver_iter);

    arma::vec V = U0;
    auto backEulerStep = [&f,&U0,&t,k](const arma::vec& x) -> arma::vec {
        arma::vec z = U0 + k * f(t.at(1),x);
        return x - z;
    };
    fsolver.fsolve(V,backEulerStep);
    U.push_back(std::move(V));

    unsigned long long i = 2;
    while (t.back() < tf) {
        double tt = _next_t(k, t.at(i-1), tf);

        V = U.at(i-1);
        auto trapStep = [&f,&U,&t,tt,i,k](const arma::vec& x) -> arma::vec {
            arma::vec z = U.at(i-1) + (k/2) * ( f(t.at(i-1),U.at(i-1)) + f(tt, x) );
            return x - z;
        };
        fsolver.fsolve(V,trapStep);

        double kk = event_handle(t.at(i-1), U.at(i-1), tt, V, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }
        t.push_back(tt);
        U.push_back(std::move(V));
        i++;
        if (kk == 0) break; // event stop
    }
    sol._prepare();
    return sol;
}

numerics::ode::ODESolution numerics::ode::am2::ode_solve(const odefunc& f, const odejacobian& jacobian, double t0, double tf, const arma::vec& U0) {
    _check_range(t0, tf);
    _check_step(t0, tf);
    double k = step;
    
    u_long m = U0.n_elem;
    ODESolution sol(m);

    std::vector<double>& t = sol._tvec;
    std::vector<arma::vec>& U = sol._Uvec;
    t.push_back(t0);
    U.push_back(U0);
    
    t.push_back(t0 + k);

    optimization::Newton fsolver(_max_solver_err, _max_solver_iter);

    arma::vec V = U0;
    auto backEulerStep = [&f,&U0,&t,k](const arma::vec& x) -> arma::vec {
        arma::vec z = U0 + k * f(t.at(1),x);
        return x - z;
    };
    auto euler_jac = [&jacobian,&U0,&t,k](const arma::vec& x) -> arma::mat {
        arma::mat J = jacobian(t.at(1),x.t());
        J = arma::eye(arma::size(J)) - k*J;
        return J;
    };
    fsolver.fsolve(V, backEulerStep, euler_jac);
    U.push_back(std::move(V));

    unsigned long long i = 2;
    while (t.back() < tf) {
        double tt = _next_t(k, t.at(i-1), tf);

        V = U.at(i-1);
        auto trapStep = [&f,&U,&t,tt,i,k](const arma::vec& x) -> arma::vec {
            arma::vec z = U.at(i-1) + (k/2) * ( f(t.at(i-1),U.at(i-1)) + f(tt, x) );
            return x - z;
        };
        auto trap_jac = [&jacobian,tt,i,k](const arma::vec& x) -> arma::mat {
            arma::mat J = jacobian(tt, x);
            J = arma::eye(arma::size(J)) - (k/2) * J;
            return J;
        };
        fsolver.fsolve(V,trapStep,trap_jac);

        double kk = event_handle(t.at(i-1), U.at(i-1), tt, V, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }
        t.push_back(tt);
        U.push_back(std::move(V));
        i++;
        if (kk == 0) break; // event stop
    }
    sol._prepare();
    return sol;
}