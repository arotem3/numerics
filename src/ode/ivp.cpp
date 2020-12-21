#include <numerics.hpp>

double numerics::ode::InitialValueProblem::_event_handle(double t1, const arma::vec& u1) {
    int num_events = events.size();

    const arma::vec& u = _prev_u.back();
    const double& t = _prev_t.back();
    double alpha = 1.0;

    for (int i=0; i < num_events; ++i) {
        const std::function<double(double, const arma::vec&)>& event = events.at(i);
        double g = event(t, u);
        double g1 = event(t1, u1);
        if (arma::sign(g) != arma::sign(g1)) { // event occurred
            if (std::abs(g1) < _event_tol) { // stopping event possibly reached
                if ((g1 < 0) and _dec.at(i)) { // {+} -> {-}
                    _stopping_event = i;
                    return 0.0;
                } else if ((g1 > 0) and _inc.at(i)) { // {-} -> {+}
                    _stopping_event = i;
                    return 0.0;
                } // if neither, then false alarm
            } else { // secant method for reducing step size.
                alpha = std::min(alpha, 0.95*std::abs(g1) / (std::abs(g1) + std::abs(g)));
            }
        }
    }
    return alpha;
}

void numerics::ode::InitialValueProblem::_check_range(double t0, arma::vec& T) {
    if (T.is_empty()) throw std::runtime_error("T array is empty, must include atleast one value.");
    T = arma::sort(T);
    if (T(0) <= t0) throw std::runtime_error("T[0] (=" + std::to_string(T(0)) + ") >= t0 (=" + std::to_string(t0) + ").");
}

void numerics::ode::InitialValueProblem::_update_solution(double& t1, arma::vec& u1, arma::vec& f1, bool full, bool is_grid_val) {
    _prev_t.push(t1); _prev_u.push(u1); _prev_f.push(f1);
    if (full or is_grid_val) {
        _t.push_back(std::move(t1));
        _U.push_back(std::move(u1));
    }
}

void numerics::ode::InitialValueProblem::_solve(const odefunc& f, const odejacobian* J, double t0, arma::vec T, const arma::vec& U0, bool full) {
    _check_range(t0,T);
    double k = _initial_step_size();

    _t.clear(); _U.clear();
    _t.push_back(t0);
    _U.push_back(U0);

    _prev_t.push(t0);
    _prev_u.push(U0);
    _prev_f.push( f(t0,U0) );
    double t1; arma::vec u1, f1;

    for (const double& tf : T) {
        while (_prev_t.back() < tf) {
            double tt = _prev_t.back();

            if (std::abs((tf - tt) - k) < 1e-10) k = tf - tt;
            else k = std::min(k, tf - tt);

            while (true) {
                double k1 = _step(k, t1, u1, f1, f, J);
                double alpha = _event_handle(t1, u1);
                if ((0 < alpha) and (alpha < 1)) k *= alpha;
                else {
                    bool is_grid_val = (std::abs(tf - tt) < 1e-10);
                    _update_solution(t1, u1, f1, full, is_grid_val);
                    k = k1;

                    if (alpha == 0) return;
                    break;
                }
            }
        }
    }
}

numerics::ode::InitialValueProblem::InitialValueProblem(double event_tol, int n_step) : _prev_u(n_step), _prev_f(n_step), _prev_t(n_step), stopping_event(_stopping_event), t(_t), U(_U) {
    _stopping_event = -1;
    if (event_tol <= 0) throw std::invalid_argument("event_tol (=" + std::to_string(event_tol) + ") must be positive.");
    _event_tol = event_tol;
}

void numerics::ode::InitialValueProblem::add_stopping_event(const std::function<double(double,const arma::vec&)>& event, const std::string& dir) {
    if (dir == "all") {
        events.push_back(event);
        _inc.push_back(true);
        _dec.push_back(true);
    } else if (dir == "inc") {
        events.push_back(event);
        _inc.push_back(true);
        _dec.push_back(false);
    } else if (dir == "dec") {
        events.push_back(event);
        _inc.push_back(false);
        _dec.push_back(true);
    }  else throw std::invalid_argument("dir (='" + dir + "') must be one of {'all','inc','dec'}.");
}

void numerics::ode::InitialValueProblem::solve_ivp(const odefunc& f, double t0, double tf, const arma::vec& U0, bool full) {
    if (U0.n_elem < 200) solver = std::make_unique<numerics::optimization::Broyden>(_solver_xtol, _solver_ftol, _solver_miter, false);
    else solver = std::make_unique<numerics::optimization::Newton>(_solver_xtol, _solver_ftol, _solver_miter, false);
    _solve(f, nullptr, t0, arma::vec({tf}), U0, full);
}

void numerics::ode::InitialValueProblem::solve_ivp(const odefunc& f, double t0, arma::vec T, const arma::vec& U0, bool full) {
    if (U0.n_elem < 200) solver = std::make_unique<numerics::optimization::Broyden>(_solver_xtol, _solver_ftol, _solver_miter, false);
    else solver = std::make_unique<numerics::optimization::Newton>(_solver_xtol, _solver_ftol, _solver_miter, false);
    _solve(f, nullptr, t0, T, U0, full);
}

void numerics::ode::InitialValueProblem::solve_ivp(const odefunc& f, const odejacobian& J, double t0, double tf, const arma::vec& U0, bool full) {
    solver = std::make_unique<numerics::optimization::Newton>(_solver_xtol, _solver_ftol, _solver_miter, false);
    _solve(f, &J, t0, arma::vec({tf}), U0, full);
}

void numerics::ode::InitialValueProblem::solve_ivp(const odefunc& f, const odejacobian& J, double t0, arma::vec T, const arma::vec& U0, bool full) {
    solver = std::make_unique<numerics::optimization::Newton>(_solver_xtol, _solver_ftol, _solver_miter, false);
    _solve(f, &J, t0, T, U0, full);
}

void numerics::ode::InitialValueProblem::as_mat(arma::vec& tt, arma::mat& uu) {
    uint n = U.size();
    uint m = U.front().n_elem;
    uu.set_size(m,n);
    tt.set_size(n);
    for (uint i=0; i < n; ++i) {
        tt(i) = t.at(i);
        uu.col(i) = U.at(i);
    }
}