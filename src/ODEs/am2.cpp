#include "ODE.hpp"

/* AM2 : multivariate Adams-Multon O(k^2) method for any explicit first order system of ODEs.
 * our equations are of the form u' = f(t,u) [u must be a row vector].
 * --- f  : f(t,u) [t must be the first variable, u the second].
 * --- t  : vector to store t-values initialized at {t_initial, t_final}.
 * --- U  : vector to store the solution first row must be u(t0).
 * --- opts : solver options. */
void ODE::am2(const odefun& f, arma::vec& t, arma::mat& U, ivp_options& opts) {
    numerics::nonlin_opts roots_opts;
    roots_opts.max_iter = implicit_ode_max_iter;
    roots_opts.err = implicit_err;
    roots_opts.use_FD_jacobian = true;
    roots_opts.init_jacobian = nullptr;
    roots_opts.init_jacobian_inv = nullptr;
    
    double k = opts.step;
    
    int m = U.n_cols; // dimension of solution space
    if (m == 0) {
        std::cerr << "am2() failed: no initial condition input." << std::endl;
        return;
    }

    arma::rowvec U0 = U.row(0);

    double t0 = t(0);
    double tf = t(1);
    t = arma::zeros(20);
    t(0) = t0;
    t(1) = t0 + k;

    U = arma::zeros(20,m);
    U.row(0) = U0;
    auto backEulerStep = [&f,&U0,&t,k](const arma::vec& x) -> arma::vec {
        arma::rowvec z = U0 + k * f(t(1),x.t());
        return x - z.t();
    };
    arma::vec V = U0.t();
    if (opts.ode_jacobian != nullptr) {
        auto jac_func = [&](const arma::vec& x) -> arma::mat {
            int nn = x.n_elem;
            arma::mat J = arma::eye(nn,nn) - k * opts.ode_jacobian->operator()(t(1), x.t());
            return J;
        };
        numerics::newton(backEulerStep, jac_func, V, roots_opts);
        opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
    } else {
        numerics::broyd(backEulerStep,V,roots_opts);
        opts.num_FD_approx_needed += roots_opts.num_FD_approx_needed;
        opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
    }
    U.row(1) = V.t();

    unsigned short i = 2;
    while (t(i) <= tf) {
        t(i) = t(i-1) + k;
        if (t(i) > tf) {
            t(i) = tf;
            k = tf - t(i);
        }
        auto trapStep = [&f,&U,&t,i,k](const arma::vec& x) -> arma::vec {
            arma::rowvec z = U.row(i-1) + (k/2) * ( f(t(i-1),U.row(i-1)) + f(t(i), x.t()) );
            return x - z.t();
        };
        V = U.row(i-1).t();
        if (opts.ode_jacobian != nullptr) {
            auto jac_func = [&](const arma::vec& x) -> arma::mat {
                int nn = x.n_elem;
                arma::mat J = arma::eye(nn,nn) - (k/2) * opts.ode_jacobian->operator()(t(i), x.t());
                return J;
            };
            numerics::newton(trapStep, jac_func, V, roots_opts);
            opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
        } else {
            numerics::broyd(trapStep,V,roots_opts);
            opts.num_FD_approx_needed += roots_opts.num_FD_approx_needed;
            opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
        }

        double kk = event_handle(opts, t(i-1), U.row(i-1), t(i), V.t(), k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        U.row(i) = V.t();
        if (t(i) == tf) break; // t span stop
        if (kk == 0) break; // event stop

        i++;
        if (i+1 == t.n_rows) {
            t = arma::join_cols(t, arma::zeros(arma::size(t)) ); // double storage
            U = arma::join_cols(U, arma::zeros(arma::size(U)) );
        }
    }
    t = t( arma::span(0,i) );
    U = U.rows( arma::span(0,i) );
}

/* AM2 : multivariate Adams-Multon O(k^2) method for any explicit first order system of ODEs.
 * our equations are of the form u' = f(t,u) [u must be a row vector].
 * --- f  : f(t,u) [t must be the first variable, u the second].
 * --- t  : vector to store t-values initialized at {t_initial, t_final}.
 * --- U  : vector to store the solution first row must be u(t0). */
ODE::ivp_options ODE::am2(const odefun& f, arma::vec& t, arma::mat& U) {
    ivp_options opts;
    opts.step = std::max(std::abs(t(1)-t(0))/100, 1e-2 );
    opts.max_nonlin_err = implicit_err;
    opts.max_nonlin_iter = implicit_ode_max_iter;
    am2(f,t,U,opts);
    return opts;
}

/* AM2 : Adams-Multon O(k^2) method for any explicit first order system of ODEs.
 * our equations are of the form u' = f(t,u).
 * --- f  : f(t,u) [t must be the first variable, u the second].
 * --- t  : vector to store t-values initialized at {t_initial, t_final}.
 * --- U0  : initial value u(t0).
 * --- opts : solver options. */
arma::vec ODE::am2(std::function<double(double,double)> f, arma::vec& t, double U0, ivp_options& opts) {
    double k = opts.step;
    
    int n; // number of points to return
    
    t = arma::regspace(t(0), k, t(1));
    n = t.n_elem;
    if (n <= 2) {
        std::cerr << "am2() failed: k is too large for the given interval." << std::endl;
        return {NAN};
    }

    arma::vec U = arma::zeros(n);
    U(0) = U0;
    auto backEulerStep = [&f,U0,&t,k](double x) -> double {
        double z = U0 + k * f(t(1),x);
        return x - z;
    };
    U(1) = numerics::secant(backEulerStep, U0, opts.max_nonlin_err);
    
    for (int i(2); i < n; ++i) {
        auto trapStep = [&f,&U,&t,i,k](double x) -> double {
            double z = U(i-1) + (k/2) * ( f(t(i-1), U(i-1)) + f(t(i), x) );
            return x - z;
        };
        U(i) = numerics::secant(trapStep, U(i-1), opts.max_nonlin_err);
    }

    return U;
}

/* AM2 : Adams-Multon O(k^2) method for any explicit first order system of ODEs.
 * our equations are of the form u' = f(t,u).
 * --- f  : f(t,u) [t must be the first variable, u the second].
 * --- t  : vector to store t-values initialized at {t_initial, t_final}.
 * --- U0  : initial value u(t0). */
arma::vec ODE::am2(std::function<double(double,double)> f, arma::vec& t, double U0) {
    ivp_options opts;
    opts.step = std::max(std::abs(t(1)-t(0))/100, 1e-2 );
    opts.max_nonlin_err = implicit_err;
    return am2(f,t,U0,opts);
}