#include <numerics.hpp>

numerics::ode::ODESolution numerics::ode::rk4::ode_solve(const odefunc& f, double t0, double tf, const arma::vec& U0) {
    _check_range(t0, tf);
    _check_step(t0, tf);
    double k = _step;

    u_long m = U0.n_elem;
    ODESolution sol(m);

    std::vector<double>& t = sol._tvec;
    std::vector<arma::vec>& U = sol._Uvec;
    t.push_back(t0);
    U.push_back(U0);

    arma::vec k1, k2, k3, k4;

    unsigned long long i = 1;
    while (t.back() < tf) {
        double tt = _next_t(k, t.at(i-1), tf);
        
        k1 = k * f(tt, U.at(i-1));
        k2 = k * f(tt + k/2, U.at(i-1) + k1/2);
        k3 = k * f(tt + k/2, U.at(i-1) + k2/2);
        k4 = k * f(tt, U.at(i-1) + k3);
        arma::vec rk4 = U.at(i-1) + (k1 + 2*k2 + 2*k3 + k4)/6;
        
        double kk = event_handle(t.at(i-1), U.at(i-1), tt, rk4, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        U.push_back(std::move(rk4));
        t.push_back(tt);
        ++i;
        if (kk == 0) break; // event stop
    }
    sol._prepare();
    return sol;
}