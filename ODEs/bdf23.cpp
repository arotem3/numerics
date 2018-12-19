#include "ODE.hpp"

void ODE::bdf23(odefun f, arma::vec& t, arma::mat& U, ivp_options& opts) {
    numerics::nonlin_opts roots_opts;
    roots_opts.max_iter = opts.max_nonlin_iter;
    roots_opts.err = opts.max_nonlin_err;
    roots_opts.use_FD_jacobian = true;
    roots_opts.init_jacobian = nullptr;
    roots_opts.init_jacobian_inv = nullptr;

    opts.num_FD_approx_needed = 0;
    opts.num_nonlin_iters_returned = 0;
    // (0.a) set up variables
    double t0 = t(0);
    double tf = t(1);
    double k = std::max( (tf - t0)/100, 1e-3);

    t = arma::zeros(20);
    t(0) = t0;
    t(1) = t0 + k;
    arma::rowvec U_temp = U.row(0);
    U = arma::zeros(20, U_temp.n_cols);
    U.row(0) = U_temp;

    // (0.b) take first step as needed for the multistep method
    auto tr = [&](const arma::vec& u) -> arma::vec {
        arma::rowvec z = U.row(0) + (k/4)*(f(t(0),U.row(0)) + f(t(0)+k/2, u.t()));
        return (u - z.t());
    };
    
    arma::vec Ustar = U.row(0).t();
    if (opts.ode_jacobian != nullptr) { // jacobian of f given exactly by user
        auto jac_func = [&t,&k,&opts](const arma::vec& u) -> arma::mat {
            int nn = u.n_elem;
            arma::mat J = arma::eye(nn,nn) - (k/4) * opts.ode_jacobian->operator()(t(0)+k/2,u.t());
            return J;
        };
        numerics::newton(tr, jac_func, Ustar, roots_opts);
        opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
    } else { // need to approximate jacobian in root finding
        numerics::broyd(tr, Ustar, roots_opts);
        opts.num_FD_approx_needed += roots_opts.num_FD_approx_needed;
        opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
    }

    auto bdf = [Ustar,k,&t,&U,&f](const arma::vec& u) -> arma::vec {
        arma::rowvec z = (4*Ustar.t() - U.row(0) + k*f(t(1),u.t()))/3.0;
        return u - z.t();
    };
    if (opts.ode_jacobian != nullptr) {
        auto jac_func = [&t,&k,&opts](const arma::vec& u) -> arma::mat {
            int nn = u.n_elem;
            arma::mat J = arma::eye(nn,nn) - (k/3) * opts.ode_jacobian->operator()(t(1),u.t());
            return J;
        };
        numerics::newton(bdf, jac_func, Ustar, roots_opts);
        opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
    } else {
        numerics::broyd(bdf, Ustar, roots_opts);
        opts.num_FD_approx_needed += roots_opts.num_FD_approx_needed;
        opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
    }

    U.row(1) = Ustar.t();

    arma::mat P;
    arma::rowvec V1,V2,Un_half,Un_full;
    unsigned short i = 1;
    bool done = false;
    while (!done) {
        double t_temp = t(i) + k;

        // (1) interpolate
            int j;
            for (j=i; j >= 0; --j) {
                if (std::abs(t(j) - t_temp) >= 2*k) { // minimum points required for interpolation
                    break;
                }
            }
            P = numerics::LPM(t(arma::span(j,i)), U.rows(arma::span(j,i)), {t_temp-k, t_temp-2*k}); // lagrange interpolation
            Un_half = P.row(0); // ~ U(n) needed for U* and V1 calculations
            Un_full = P.row(1); // ~ U(n-1) needed for V2 calculation
        
        // (2) approximate the ODEs
            auto tr = [&](const arma::vec& u) -> arma::vec {
                arma::rowvec z = Un_half + (k/4)*(f(t_temp-k, Un_half) + f(t_temp-k/2.0, u.t()));
                return (u - z.t());
            };
            Ustar = Un_half.t();
            if (opts.ode_jacobian != nullptr) {
                auto jac_func = [&t,&k,&opts,&t_temp](const arma::vec& u) -> arma::mat {
                    int nn = u.n_elem;
                    arma::mat J = arma::eye(nn,nn) - (k/4) * opts.ode_jacobian->operator()(t_temp-k/2,u.t());
                    return J;
                };
                numerics::newton(tr, jac_func, Ustar, roots_opts);
                opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
            } else {
                numerics::broyd(tr, Ustar, roots_opts);
                opts.num_FD_approx_needed += roots_opts.num_FD_approx_needed;
                opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
            }

            auto bdf = [Ustar,k,t_temp,&f,&Un_half](const arma::vec& u) -> arma::vec {
                arma::rowvec z = (4*Ustar.t() - Un_half + k*f(t_temp,u.t()))/3.0;
                return (u - z.t());
            };
            if (opts.ode_jacobian != nullptr) {
                auto jac_func = [&t,&k,&opts,&t_temp](const arma::vec& u) -> arma::mat {
                    int nn = u.n_elem;
                    arma::mat J = arma::eye(nn,nn) - (k/3) * opts.ode_jacobian->operator()(t_temp,u.t());
                    return J;
                };
                numerics::newton(bdf, jac_func, Ustar, roots_opts);
                opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
            } else {
                numerics::broyd(bdf, Ustar, roots_opts);
                opts.num_FD_approx_needed += roots_opts.num_FD_approx_needed;
                opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
            }
            V1 = Ustar.t();

            auto bdf_full = [&](const arma::vec& u) -> arma::vec {
                arma::rowvec z = (4*Un_half - Un_full + 2*k*f(t_temp,u.t()))/3.0;
                return (u - z.t());
            };
            Ustar = Un_half.t();
            if (opts.ode_jacobian != nullptr) {
                auto jac_func = [&t,&k,&opts,&t_temp](const arma::vec& u) -> arma::mat {
                    int nn = u.n_elem;
                    arma::mat J = arma::eye(nn,nn) - (2*k/3) * opts.ode_jacobian->operator()(t_temp,u.t());
                    return J;
                };
                numerics::newton(bdf_full, jac_func, Ustar, roots_opts);
                opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
            } else {
                numerics::broyd(bdf_full, Ustar, roots_opts);
                opts.num_FD_approx_needed += roots_opts.num_FD_approx_needed;
                opts.num_nonlin_iters_returned += roots_opts.num_iters_returned;
            }
            V2 = Ustar.t();

        // (3) step size adjustment
            double R = arma::norm(V1 - V2, "Inf");
            double Q = std::sqrt(opts.adaptive_max_err/R);

            double kk = event_handle(opts,t(i), U.row(i), t_temp, V1, k); // new k based on events
            if (R < opts.adaptive_max_err) {
                if (0 < kk && kk < k) {     // event require us to try again with a smaller step size;
                    k = kk;
                    continue;
                }

                t(i+1) = t_temp;
                U.row(i+1) = V1;

                if (kk == 0) break;         // event requires us to stop
                i++;
                if (i+1 == t.n_rows) {
                    t = arma::join_cols(t, arma::zeros(arma::size(t)) ); // double storage
                    U = arma::join_cols(U, arma::zeros(arma::size(U)) );
                }
            }

            if (Q > rk45_qmax) k *= rk45_qmax;
            else if (Q < rk45_qmin) k *= rk45_qmin;
            else k *= Q;

            if (k < opts.adaptive_step_min) {
                std::cerr << "bdf23() error: could not converge with in the required tolerance." << std::endl;
                return;
            }
            if (k > opts.adaptive_step_max) k = opts.adaptive_step_max;

            if (t_temp + k > tf) k = tf - t_temp;
            
            if (t_temp - 2*k <= t0) k = (t_temp - t0)/2.1;

            if (t_temp >= tf) done = true;
    }
    t = t( arma::span(0,i+1) );
    U = U.rows( arma::span(0,i+1) );
}

ODE::ivp_options ODE::bdf23(odefun f, arma::vec& t, arma::mat& U) {
    ivp_options opts;
    opts.adaptive_max_err = 1e-4;
    opts.adaptive_step_max = rk45_kmax;
    opts.adaptive_step_min = rk45_kmin;
    opts.max_nonlin_err = implicit_err;
    opts.max_nonlin_iter = implicit_ode_max_iter;
    bdf23(f,t,U,opts);
    return opts;
}

arma::vec ODE::bdf23(std::function<double(double,double)> f, arma::vec& t, double U0, ivp_options& opts) {
    // (0.a) set up variables
    double t0 = t(0);
    double tf = t(1);
    double k = std::max( (tf - t0)/100, 1e-3);

    t(1) = t(0) + k;
    arma::vec U = {U0};

    // (0.b) take first step as needed for the multistep method
    auto tr = [&](double u) -> double {
        double z = U(0) + (k/4)*(f(t(0),U(0)) + f(t(0)+k/2, u));
        return u - z;
    };
    double Ustar = U(0);
    Ustar = numerics::secant(tr, Ustar, opts.max_nonlin_err);

    auto bdf = [&](double u) -> double {
        double z = (4*Ustar - U(0) + k*f(t(1),u))/3.0;
        return u - z;
    };
    U(1) = numerics::secant(bdf, Ustar, opts.max_nonlin_err);

    arma::vec P;
    double V1,V2;
    int i = 1;
    bool done = false;
    while (!done) {
        double t_temp = t(i-1) + k;

        // (1) interpolate
        int j;
        for (j=i; j > 0; --j) {
            if (std::abs(t(j) - t(i)) >= 2*k) { // minimum points required for interpolation
                break;
            }
        }
        P = numerics::lagrangeInterp(t(arma::span(j,i)), U(arma::span(j,i)), {t_temp-k, t_temp-2*k}); // lagrange interpolation
        double Un_half = P(0); // ~ U(n) needed for U* and V1 calculations
        double Un_full = P(1); // ~ U(n-1) needed for V2 calculation

        // (2) approximate the ODEs
        auto tr = [&](double u) -> double {
            double z = Un_half + (k/4)*(f(t_temp-k, Un_half) + f(t_temp-k/2, u));
            return u - z;
        };
        Ustar = numerics::secant(tr, Un_half, opts.max_nonlin_err);

        auto bdf = [&](double u) -> double {
            double z = (4*Ustar - Un_half + k*f(t_temp,u))/3.0;
            return u - z;
        };
        V1 = numerics::secant(bdf, Ustar, opts.max_nonlin_err);

        auto bdf_full = [&](double u) -> double {
            double z = (4*Un_half - Un_full + 2*k*f(t_temp,u))/3.0;
            return u - z;
        };
        V2 = numerics::secant(bdf_full, Un_half, opts.max_nonlin_err);

        // (3) step size adjustment
        double R = std::abs(V1 -V2);
        double Q = std::sqrt(opts.adaptive_max_err/R);

        if (R < opts.adaptive_max_err) {
            arma::rowvec t_next = {t_temp};
            arma::rowvec U_next = {V1};
            t.insert_rows(i+1, t_next);
            U.insert_rows(i+1, U_next);
            i++;
        }

        if (Q > rk45_qmax) k *= rk45_qmax;
        else if (Q < rk45_qmin) k *= rk45_qmin;
        else k *= Q;

        if (k < opts.adaptive_step_min) {
            std::cerr << "bdf23() error: could not converge with in the required tolerance." << std::endl;
            return U;
        }
        if (k > opts.adaptive_step_max) k = opts.adaptive_step_max;

        if (t_temp + k > tf) k = tf - t_temp;
         
        if (t_temp - 2*k < t0) k = (t_temp - t0)/2;

        if (t_temp >= tf) done = true;
    }
    return U;
}

arma::vec ODE::bdf23(std::function<double(double,double)> f, arma::vec& t, double U0) {
    ivp_options opts;
    opts.adaptive_max_err = 1e-4;
    opts.adaptive_step_max = rk45_kmax;
    opts.adaptive_step_min = rk45_kmin;
    opts.max_nonlin_err = implicit_err;
    return bdf23(f,t,U0,opts);
}