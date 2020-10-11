#include <numerics.hpp>

void numerics::ode::rk5i::solve_ivp(const odefunc& f, double t0, double tf, const arma::vec& U0) {
    _check_range(t0, tf);
    _check_step(t0, tf);
    double k = _step;

    _t.clear(); _U.clear();
    _t.push_back(t0);
    _U.push_back(U0);

    arma::vec V1, V2, V3, V4;

    optimization::Broyd fsolver(_max_solver_err, _max_solver_iter);

    unsigned long long i = 1;
    while (_t.back() < tf) {
        double tt = _next_t(k, _t.at(i-1), tf);

        V1 = f(_t.at(i-1), _U.at(i-1));

        arma::vec V_temp = _U.at(i-1);
        auto v2f = [this,&f,&V1,i,k](const arma::vec& x) -> arma::vec {
            arma::vec z = f(_t.at(i-1), _U.at(i-1) + 0.125*k*V1 + 0.125*k*x);
            return x - z;
        };
        fsolver.fsolve(V_temp, v2f);

        V2 = V_temp;
        auto v3f = [this,&f,&V1,&V2,i,k](const arma::vec& x) -> arma::vec {
            arma::vec z = f(_t.at(i-1) + 0.7*k, _U.at(i-1) - 0.01*k*V1 + 0.56*k*V2 + 0.15*k*x);
            return x - z;
        };
        fsolver.fsolve(V_temp, v3f);
        V3 = V_temp;
    
        V4 = f(tt, _U.at(i-1) + (2.0/7)*k*V1 + (5.0/7)*k*V3);

        arma::vec rk5 = _U.at(i-1) + k * ( (1.0/14)*V1 + (32.0/81)*V2 + (250.0/567)*V3 + (5.0/54)*V4 );
        
        double kk = event_handle(_t.at(i-1), _U.at(i-1), tt, rk5, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        _U.push_back(std::move(rk5));
        _t.push_back(tt);
        if (kk == 0) break; // event stop
        i++;
    }
}

void numerics::ode::rk5i::solve_ivp(const odefunc& f, const odejacobian& jacobian, double t0, double tf, const arma::vec& U0) {
    _check_range(t0, tf);
    _check_step(t0, tf);
    double k = _step;

    _t.clear(); _U.clear();
    _t.push_back(t0);
    _U.push_back(U0);

    arma::vec V1, V2, V3, V4;

    optimization::Newton fsolver(_max_solver_err, _max_solver_iter);

    unsigned long long i = 1;
    while (_t.back() < tf) {
        double tt = _next_t(k, _t.at(i-1), tf);

        V1 = f(_t.at(i-1), _U.at(i-1));

        arma::vec V_temp = _U.at(i-1);
        auto v2f = [this,&f,&V1,i,k](const arma::vec& x) -> arma::vec {
            arma::vec z = f(_t.at(i-1), _U.at(i-1) + 0.125*k*V1 + 0.125*k*x);
            return x - z;
        };
        auto v2f_jac = [this,&jacobian,&V1,i,k](const arma::vec& x) -> arma::mat {
            arma::mat J = jacobian(_t.at(i-1), _U.at(i-1) + 0.125*k*V1 + 0.125*k*x);
            J = arma::eye(arma::size(J)) - 0.125*k*J;
            return J;
        };
        fsolver.fsolve(V_temp, v2f, v2f_jac);

        V2 = V_temp;
        auto v3f = [this,&f,&V1,&V2,i,k](const arma::vec& x) -> arma::vec {
            arma::vec z = f(_t.at(i-1) + 0.7*k, _U.at(i-1) - 0.01*k*V1 + 0.56*k*V2 + 0.15*k*x);
            return x - z;
        };
        auto v3f_jac = [this,&jacobian,&V1,&V2,i,k](const arma::vec& x) -> arma::mat {
            arma::mat J = jacobian(_t.at(i-1) + 0.7*k, _U.at(i-1) - 0.01*k*V1 + 0.56*k*V2 + 0.15*k*x);
            J = arma::eye(arma::size(J)) - 0.15*k*J;
            return J;
        };
        fsolver.fsolve(V_temp, v3f, v3f_jac);
        V3 = V_temp;
    
        V4 = f(tt, _U.at(i-1) + (2.0/7)*k*V1 + (5.0/7)*k*V3);

        arma::vec rk5 = _U.at(i-1) + k * ( (1.0/14)*V1 + (32.0/81)*V2 + (250.0/567)*V3 + (5.0/54)*V4 );
        
        double kk = event_handle(_t.at(i-1), _U.at(i-1), tt, rk5, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        _U.push_back(std::move(rk5));
        _t.push_back(tt);
        if (kk == 0) break; // event stop
        i++;
    }
}