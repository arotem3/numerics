#include "ODE.hpp"

//--- multivariate RK4 method for any explicit first order system of ODEs ---//
//--- our equations are of the form u' = f(t,u) [u must be a row vector] ----//
//----- f  : f(t,U) [t must be the first variable, U the second] ------------//
//----- t  : vector to store t-values initialized at {t0, tf} ---------------//
//----- U  : vector to store the solution first row must be y0 --------------//
//----- k  : t spacing i.e. |t(i+1) - t(i)| this method is O(k^4) -----------//
void ODE::rk4(const odefun& f, arma::vec& t, arma::mat& U, ivp_options& opts) {
    double k = opts.step;
    int m = U.n_cols; // dimension of solution space
    if (m == 0) { // no initial conditions err
        std::cerr << "rk4() failed: no initial condition input." << std::endl;
        return;
    }
    arma::rowvec U0 = U.row(0); 

    double t0 = t(0);
    double tf = t(1);
    t = arma::zeros(20);
    t(0) = t0;

    U = arma::zeros(20,m);
    U.row(0) = U0;

    arma::rowvec k1;
    arma::rowvec k2;
    arma::rowvec k3;
    arma::rowvec k4;

    unsigned short i = 1;
    while (t(i) <= tf) {
        t(i) = t(i-1) + k;
        if (t(i) > tf) {
            t(i) = tf;
            k = tf - t(i-1);
        }
        k1 = k * f(t(i-1), U.row(i-1));
        k2 = k * f(t(i-1) + k/2, U.row(i-1) + k1/2);
        k3 = k * f(t(i-1) + k/2, U.row(i-1) + k2/2);
        k4 = k * f(t(i-1), U.row(i-1) + k3);
        arma::rowvec rk4 = U.row(i-1) + (k1 + 2*k2 + 2*k3 + k4)/6;
        
        double kk = event_handle(opts, t(i-1), U.row(i-1), t(i), rk4, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        U.row(i) = rk4;
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

ODE::ivp_options ODE::rk4(const odefun& f, arma::vec& t, arma::mat& U) {
    ivp_options opts;
    opts.step = std::max(std::abs(t(1)-t(0))/100, 1e-2 );
    rk4(f,t,U,opts);
    return opts;
}


//--- one dimensional version ---//
arma::vec ODE::rk4(std::function<double(double,double)> f, arma::vec& t, double U0, ivp_options& opts) {
    double k = opts.step;
    t = arma::regspace(t(0), k, t(1));
    int n = t.n_elem; // number of points to return
    if (n <= 2) {
        std::cerr << "rk4() failed: k too small for given interval." << std::endl;
        return {NAN};
    }
    
    arma::vec U = arma::zeros(n);
    U(0) = U0;

    double k1, k2, k3, k4;

    for (int i(1); i < n; ++i) {
        k1 = k * f(t(i-1), U(i-1));
        k2 = k * f(t(i-1) + k/2, U(i-1) + k1/2);
        k3 = k * f(t(i-1) + k/2, U(i-1) + k2/2);
        k4 = k * f(t(i-1), U(i-1) + k3);
        U(i) = U(i-1) + (k1 + 2*k2 + 2*k3 + k4)/6;
    }

    return U;
}

arma::vec ODE::rk4(std::function<double(double,double)> f, arma::vec& t, double U0) {
    ivp_options opts;
    opts.step = std::max(std::abs(t(1)-t(0))/100, 1e-2 );
    return rk4(f,t,U0,opts);
}