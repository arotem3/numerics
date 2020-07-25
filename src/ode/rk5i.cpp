#include <numerics.hpp>

numerics::ode::ODESolution numerics::ode::rk5i::ode_solve(const odefunc& f, double t0, double tf, const arma::vec& U0) {
    _check_range(t0, tf);
    _check_step(t0, tf);
    double k = _step;

    u_long m = U0.n_elem;
    ODESolution sol(m);

    std::vector<double>& t = sol._tvec;
    std::vector<arma::vec>& U = sol._Uvec;
    t.push_back(t0);
    U.push_back(U0);

    arma::vec V1, V2, V3, V4;

    optimization::Broyd fsolver(_max_solver_err, _max_solver_iter);

    unsigned long long i = 1;
    while (t.back() < tf) {
        double tt = _next_t(k, t.at(i-1), tf);

        V1 = f(t.at(i-1), U.at(i-1));

        arma::vec V_temp = U.at(i-1);
        auto v2f = [&f,&U,&t,&V1,i,k](const arma::vec& x) -> arma::vec {
            arma::vec z = f(t.at(i-1), U.at(i-1) + 0.125*k*V1 + 0.125*k*x);
            return x - z;
        };
        fsolver.fsolve(V_temp, v2f);

        V2 = V_temp;
        auto v3f = [&f,&U,&t,&V1,&V2,i,k](const arma::vec& x) -> arma::vec {
            arma::vec z = f(t.at(i-1) + 0.7*k, U.at(i-1) - 0.01*k*V1 + 0.56*k*V2 + 0.15*k*x);
            return x - z;
        };
        fsolver.fsolve(V_temp, v3f);
        V3 = V_temp;
    
        V4 = f(tt, U.at(i-1) + (2.0/7)*k*V1 + (5.0/7)*k*V3);

        arma::vec rk5 = U.at(i-1) + k * ( (1.0/14)*V1 + (32.0/81)*V2 + (250.0/567)*V3 + (5.0/54)*V4 );
        
        double kk = event_handle(t.at(i-1), U.at(i-1), tt, rk5, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        U.push_back(std::move(rk5));
        t.push_back(tt);
        if (kk == 0) break; // event stop
        i++;
    }
    sol._prepare();
    return sol;
}

numerics::ode::ODESolution numerics::ode::rk5i::ode_solve(const odefunc& f, const odejacobian& jacobian, double t0, double tf, const arma::vec& U0) {
    _check_range(t0, tf);
    _check_step(t0, tf);
    double k = _step;

    u_long m = U0.n_elem;
    ODESolution sol(m);

    std::vector<double>& t = sol._tvec;
    std::vector<arma::vec>& U = sol._Uvec;
    t.push_back(t0);
    U.push_back(U0);

    arma::vec V1, V2, V3, V4;

    optimization::Newton fsolver(_max_solver_err, _max_solver_iter);

    unsigned long long i = 1;
    while (t.back() < tf) {
        double tt = _next_t(k, t.at(i-1), tf);

        V1 = f(t.at(i-1), U.at(i-1));

        arma::vec V_temp = U.at(i-1);
        auto v2f = [&f,&U,&t,&V1,i,k](const arma::vec& x) -> arma::vec {
            arma::vec z = f(t.at(i-1), U.at(i-1) + 0.125*k*V1 + 0.125*k*x);
            return x - z;
        };
        auto v2f_jac = [&jacobian,&U,&t,&V1,i,k](const arma::vec& x) -> arma::mat {
            arma::mat J = jacobian(t.at(i-1), U.at(i-1) + 0.125*k*V1 + 0.125*k*x);
            J = arma::eye(arma::size(J)) - 0.125*k*J;
            return J;
        };
        fsolver.fsolve(V_temp, v2f, v2f_jac);

        V2 = V_temp;
        auto v3f = [&f,&U,&t,&V1,&V2,i,k](const arma::vec& x) -> arma::vec {
            arma::vec z = f(t.at(i-1) + 0.7*k, U.at(i-1) - 0.01*k*V1 + 0.56*k*V2 + 0.15*k*x);
            return x - z;
        };
        auto v3f_jac = [&jacobian,&U,&t,&V1,&V2,i,k](const arma::vec& x) -> arma::mat {
            arma::mat J = jacobian(t.at(i-1) + 0.7*k, U.at(i-1) - 0.01*k*V1 + 0.56*k*V2 + 0.15*k*x);
            J = arma::eye(arma::size(J)) - 0.15*k*J;
            return J;
        };
        fsolver.fsolve(V_temp, v3f, v3f_jac);
        V3 = V_temp;
    
        V4 = f(tt, U.at(i-1) + (2.0/7)*k*V1 + (5.0/7)*k*V3);

        arma::vec rk5 = U.at(i-1) + k * ( (1.0/14)*V1 + (32.0/81)*V2 + (250.0/567)*V3 + (5.0/54)*V4 );
        
        double kk = event_handle(t.at(i-1), U.at(i-1), tt, rk5, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        U.push_back(std::move(rk5));
        t.push_back(tt);
        if (kk == 0) break; // event stop
        i++;
    }
    sol._prepare();
    return sol;
}