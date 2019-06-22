#include <numerics.hpp>

/* ode_solve(f, t, U) : runge kutta O(K^4) method for any explicit first order system of ODEs.
 * our equations are of the form u' = f(t,u) [u must be a row vector].
 * --- f  : f(t,u) [t must be the first variable, u the second].
 * --- t  : vector to store t-values initialized at {t_initial, t_final}.
 * --- U  : vector to store the solution first row must be u(t0). */
void numerics::ode::rk4::ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U) {
    double k = step;
    arma::rowvec U0 = arma::vectorise(U).t();
    int m = U0.n_cols; // dimension of solution space
    if (m == 0) { // no initial conditions err
        std::cerr << "rk4() failed: no initial condition input." << std::endl;
        return;
    }

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
        
        double kk = event_handle(t(i-1), U.row(i-1), t(i), rk4, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        U.row(i) = rk4;
        if (t(i) >= tf) break; // t span stop
        if (kk == 0) break; // event stop

        i++;
        if (i+1 == t.n_rows) {
            t.resize(t.n_rows*2,1);
            U.resize(U.n_rows*2,U.n_cols);
        }
    }
    t = t( arma::span(0,i) );
    U = U.rows( arma::span(0,i) );
}