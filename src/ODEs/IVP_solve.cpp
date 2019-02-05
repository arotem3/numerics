#include "ODE.hpp"

/* IVP_SOLVE : general solver with choice of method for any explicit first order system of ODEs.
 * our equations are of the form u' = f(t,u) [u must be a row vector].
 * --- f  : f(t,u) [t must be the first variable, u the second].
 * --- t  : vector to store t-values initialized at {t_initial, t_final}.
 * --- U  : vector to store the solution first row must be initial value u(t0).
 * --- opts : solver options.
 * --- solver : choice of solver. */
numerics::CubicInterp ODE::IVP_solve(const odefun& f, arma::vec& t, arma::mat& U, ivp_options& opts, ode_solver solver) {
    if (solver == RK45) {
        rk45(f,t,U,opts);
    } else if (solver == BDF23) {
        bdf23(f,t,U,opts);
    } else if (solver == RK4) {
        rk4(f,t,U,opts);
    } else if (solver == RK5I) {
        rk5i(f,t,U,opts);
    } else if (solver == AM1) {
        am1(f,t,U,opts);
    } else if (solver == AM2) {
        am2(f,t,U,opts);
    } else {
        std::cerr << "IVP_solve() error: invalid solver selection. Using rk45()" << std::endl;
        rk45(f,t,U,opts);
    }
    
    numerics::CubicInterp soln(t,U);
    return soln;
}

/* IVP_SOLVE : general solver with choice of method for any explicit first order system of ODEs.
 * our equations are of the form u' = f(t,u) [u must be a row vector].
 * --- f  : f(t,u) [t must be the first variable, u the second].
 * --- t  : vector to store t-values initialized at {t_initial, t_final}.
 * --- U  : vector to store the solution first row must be initial value u(t0).
 * --- solver : choice of solver. */
numerics::CubicInterp ODE::IVP_solve(const odefun& f, arma::vec& t, arma::mat& U, ode_solver solver) {
    ivp_options opts;
    opts.adaptive_max_err = 1e-4;
    opts.adaptive_step_max = rk45_kmax;
    opts.adaptive_step_min = rk45_kmin;
    opts.max_nonlin_err = implicit_err;
    opts.max_nonlin_iter = implicit_ode_max_iter;
    opts.step = std::max(std::abs(t(1)-t(0))/100, 1e-2 );
    return IVP_solve(f,t,U,opts,solver);
}