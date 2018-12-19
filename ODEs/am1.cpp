#include "ODE.hpp"

//--- multivariate implicit Euler method O(k) for any explicit first order system of ODEs ---//
//--- our equations are of the form u' = f(t,u) [u must be a row vector] --------------------//
//----- f  : f(t,u) [t must be the first variable, u the second] ----------------------------//
//----- t  : vector to store t-values initialized at {t0, tf} -------------------------------//
//----- U  : vector to store the solution first row must be u(t0) ---------------------------//
void ODE::am1(odefun f, arma::vec& t, arma::mat& U, ivp_options& opts) {
    numerics::nonlin_opts roots_opts;
    roots_opts.max_iter = implicit_ode_max_iter;
    roots_opts.err = implicit_err;
    roots_opts.use_FD_jacobian = true;
    roots_opts.init_jacobian = nullptr;
    roots_opts.init_jacobian_inv = nullptr;
    
    double k = opts.step;
    
    int m = U.n_cols; // dimension of solution space
    if (m == 0) { // no initial conditions err
        std::cerr << "am1() failed: no initial condition input." << std::endl;
        return;
    }

    arma::rowvec U0 = U.row(0);
    double t0 = t(0);
    double tf = t(1);
    t = arma::zeros(20);
    t(0) = t0;
    U = arma::zeros(20,m); 
    U.row(0) = U0;

    unsigned short i = 1;
    while (t(i) <= tf) {
        t(i) = t(i-1) + k;
        if (t(i) > tf) {
            t(i) = tf;
            k = tf - t(i-1);
        }
        auto eulerStep = [&f,&U,&t,i,k](const arma::vec& x) -> arma::vec {
            arma::rowvec z = x.t() - (  U.row(i-1) + k * f(t(i), x.t())  );
            return z.t();
        };
        arma::vec UU = U.row(i-1).t();
        if (opts.ode_jacobian != nullptr) {
            auto jac_func = [&](const arma::vec& x) -> arma::mat {
                int nn = x.n_elem;
                arma::mat J = arma::eye(nn,nn) - k * opts.ode_jacobian->operator()(t(i), x.t());
                return J;
            };
            numerics::newton(eulerStep, jac_func, UU, roots_opts);
            opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
        } else {
            numerics::broyd(eulerStep,UU,roots_opts);
            opts.num_FD_approx_needed += roots_opts.num_FD_approx_needed;
            opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
        }

        double kk = event_handle(opts, t(i-1), U.row(i-1), t(i), UU.t(), k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        U.row(i) = UU.t();
        if (t(i) == tf) break; // t span stop
        if (kk == 0) break; // event stop
        
        i++;                        // move to the next step
        if (i+1 == t.n_rows) {
            t = arma::join_cols(t, arma::zeros(arma::size(t)) ); // double storage
            U = arma::join_cols(U, arma::zeros(arma::size(U)) );
        }
    }
    t = t( arma::span(0,i) );
    U = U.rows( arma::span(0,i) );
}

ODE::ivp_options ODE::am1(odefun f, arma::vec& t, arma::mat& U) {
    ivp_options opts;
    opts.step = std::max(std::abs(t(1)-t(0))/100, 1e-2 );
    opts.max_nonlin_err = implicit_err;
    opts.max_nonlin_iter = implicit_ode_max_iter;
    am1(f,t,U,opts);
    return opts;
}


//--- one dimensional version ---//
arma::vec ODE::am1(std::function<double(double,double)> f, arma::vec& t, double U0, ivp_options& opts) {
    double k = opts.step;
    
    t = arma::regspace(t(0), k ,t(1));
    int n = t.n_elem; // number of points to return

    if (n <= 2) {
        std::cerr << "odeBE() failed: k too large for the given interval." << std::endl;
        return {NAN};
    }

    arma::vec U = arma::zeros(n);
    U(0) = U0;

    for (int i(1); i < n; ++i) {
        auto eulerStep = [&f,&U,&t,i,k](double x) -> double {
            return x - (U(i-1) + k * f(t(i), x));
        };
        U(i) = numerics::secant(eulerStep,U(i-1), opts.max_nonlin_err);
    }
    
    return U;
}

arma::vec ODE::am1(std::function<double(double,double)> f, arma::vec& t, double U0) {
    ivp_options opts;
    opts.step = std::max(std::abs(t(1)-t(0))/100, 1e-2 );
    opts.max_nonlin_err = implicit_err;
    return am1(f,t,U0,opts);
}