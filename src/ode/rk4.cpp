#include <numerics.hpp>

void numerics::ode::rk4::solve_ivp(const odefunc& f, double t0, double tf, const arma::vec& U0) {
    _check_range(t0, tf);
    _check_step(t0, tf);
    double k = _step;

    _t.clear(); _U.clear();
    _t.push_back(t0);
    _U.push_back(U0);

    arma::vec k1, k2, k3, k4;

    unsigned long long i = 1;
    while (_t.back() < tf) {
        double tt = _next_t(k, _t.at(i-1), tf);
        
        k1 = k * f(tt, _U.at(i-1));
        k2 = k * f(tt + k/2, _U.at(i-1) + k1/2);
        k3 = k * f(tt + k/2, _U.at(i-1) + k2/2);
        k4 = k * f(tt, _U.at(i-1) + k3);
        arma::vec rk4 = _U.at(i-1) + (k1 + 2*k2 + 2*k3 + k4)/6;
        
        double kk = event_handle(_t.at(i-1), _U.at(i-1), tt, rk4, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        _U.push_back(std::move(rk4));
        _t.push_back(tt);
        ++i;
        if (kk == 0) break; // event stop
    }
}