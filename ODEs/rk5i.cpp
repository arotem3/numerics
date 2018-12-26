#include "ODE.hpp"

//--- multivariate implicit rk method O(k^5) for any explicit first order system of ODEs ---//
//--- our equations are of the form u' = f(t,u) [u must be a row vector] -------------------//
//----- f  : f(t,u) [t must be the first variable, u the second] ---------------------------//
//----- t  : vector to store t-values initialized at {t0, tf} ------------------------------//
//----- U  : vector to store the solution first row must be y0 -----------------------------//
void ODE::rk5i(const odefun& f, arma::vec& t, arma::mat& U, ivp_options& opts) {
    numerics::nonlin_opts roots_opts;
    roots_opts.max_iter = opts.max_nonlin_iter;
    roots_opts.err = opts.max_nonlin_err;
    roots_opts.use_FD_jacobian = true;
    roots_opts.init_jacobian = nullptr;
    roots_opts.init_jacobian_inv = nullptr;

    opts.num_FD_approx_needed = 0;
    opts.num_nonlin_iters_returned = 0;
    double k = opts.step;
    
    int m = U.n_cols; // dimension of solution space
    if (m == 0) { // no initial conditions err
        std::cerr << "rk5i() failed: no initial condition input." << std::endl;
        return;
    }
    arma::rowvec U0 = U.row(0); 

    double t0 = t(0);
    double tf = t(1);
    t = arma::zeros(20);
    t(0) = t0;

    U = arma::zeros(20,m);
    U.row(0) = U0;

    arma::rowvec V1;
    arma::rowvec V2;
    arma::rowvec V3;
    arma::rowvec V4;

    unsigned short i = 1;
    while (t(i) <= tf) {
        t(i) = t(i-1) + k;
        if (t(i) > tf) {
            t(i) = tf;
            k = tf - t(i-1);
        }
        V1 = f(t(i-1), U.row(i-1));

        auto v2f = [&f,&U,&t,&V1,i,k](const arma::vec& x) -> arma::vec {
            arma::rowvec z = f(t(i-1), U.row(i-1) + 0.125*k*V1 + 0.125*k*x.t());
            return x - z.t();
        };
        arma::vec V_temp = U.row(i-1).t();
        if (opts.ode_jacobian != nullptr) {
            auto jac_func = [&](const arma::vec& x) -> arma::mat {
                int nn = x.n_elem;
                arma::mat J = arma::eye(nn,nn) - 0.125*k*opts.ode_jacobian->operator()(t(i-1), U.row(i-1) + 0.125*k*V1 + 0.125*k*x.t());
                return J;
            };
            numerics::newton(v2f, jac_func, V_temp, roots_opts);
            opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
        } else {
            numerics::broyd(v2f,V_temp,roots_opts);
            opts.num_FD_approx_needed += roots_opts.num_FD_approx_needed;
            opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
        }
        V2 = V_temp.t();

        auto v3f = [&f,&U,&t,&V1,&V2,i,k](const arma::vec& x) -> arma::vec {
            arma::rowvec z = f(t(i-1) + 0.7*k, U.row(i-1) - 0.01*k*V1 + 0.56*k*V2 + 0.15*k*x.t());
            return x - z.t();
        };
        V_temp = U.row(i-1).t();
        if (opts.ode_jacobian != nullptr) {
            auto jac_func = [&](const arma::vec& x) -> arma::mat {
                int nn = x.n_elem;
                arma::mat J = arma::eye(nn,nn) - 0.15*k*opts.ode_jacobian->operator()(t(i-1) + 0.7*k, U.row(i-1) - 0.01*k*V1 + 0.56*k*V2 + 0.15*k*x.t());
                return J;
            };
            numerics::newton(v3f, jac_func, V_temp, roots_opts);
            opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
        } else {
            numerics::broyd(v3f,V_temp,roots_opts);
            opts.num_FD_approx_needed += roots_opts.num_FD_approx_needed;
            opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
        }
        V3 = V_temp.t();

        V4 = f(t(i-1) + k, U.row(i-1) + (2.0/7)*k*V1 + (5.0/7)*k*V3);

        arma::rowvec rk5 = U.row(i-1) + k * ( (1.0/14)*V1 + (32.0/81)*V2 + (250.0/567)*V3 + (5.0/54)*V4 );
        
        double kk = event_handle(opts, t(i-1), U.row(i-1), t(i), rk5, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        U.row(i) = rk5;
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

ODE::ivp_options ODE::rk5i(const odefun& f, arma::vec& t, arma::mat& U) {
    ivp_options opts;
    opts.step = std::max(std::abs(t(1)-t(0))/100, 1e-2 );
    opts.max_nonlin_err = implicit_err;
    opts.max_nonlin_iter = implicit_ode_max_iter;
    rk5i(f,t,U,opts);
    return opts;
}

//--- one dimensional version ---//
arma::vec ODE::rk5i(std::function<double(double,double)> f, arma::vec& t, double y0, ivp_options& opts) {
    double k = opts.step;

    t = arma::regspace(t(0), k, t(1));
    int n = t.n_elem; // number of points to return
    if (n <= 2) {
        std::cerr << "rk5i() failed: k too small for given interval." << std::endl;
        return {NAN};
    }

    arma::vec U = arma::zeros(n);
    U(0) = y0;

    double V1, V2, V3, V4;

    for (int i(1); i < n; ++i) {
        V1 = f(t(i-1), U(i-1));

        auto v2f = [&f,&U,&t,&V1,i,k](double x) -> double {
            double z = f(t(i-1), U(i-1) + 0.125*k*V1 + 0.125*k*x);
            return x - z;
        };
        V2 = numerics::secant(v2f, U(i-1), opts.max_nonlin_err);

        auto v3f = [&f,&U,&t,&V1,&V2,i,k](double x) -> double {
            double z = f(t(i-1) + 0.7*k, U(i-1) - 0.01*k*V1 + 0.56*k*V2 + 0.15*k*x);
            return x - z;
        };
        V3 = numerics::secant(v3f, U(i-1), opts.max_nonlin_err);
        
        V4 = f(t(i-1) + k, U(i-1) + (2.0/7)*k*V1 + (5.0/7)*k*V3);
        U(i) = U(i-1) + k * ( (1.0/14)*V1 + (32.0/81)*V2 + (250.0/567)*V3 + (5.0/54)*V4 );
    }
    return U;
}

arma::vec ODE::rk5i(std::function<double(double,double)> f, arma::vec& t, double y0) {
    ivp_options opts;
    opts.step = std::max(std::abs(t(1)-t(0))/100, 1e-2 );
    opts.max_nonlin_err = implicit_err;
    return rk5i(f,t,y0,opts);
}