#include <numerics.hpp>

/* ode_solve(f, t, U) : multivariate implicit Euler method O(k) for any explicit first order system of ODEs.
 * our equations are of the form u' = f(t,u) [u must be a row vector].
 * --- f  : f(t,u) [t must be the first variable, u the second].
 * --- t  : vector to store t-values initialized at {t_initial, t_final}.
 * --- U  : vector to store the solution first row must be u(t0). */
void numerics::ode::am1::ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U) {
    double k = step;

    arma::rowvec U0 = arma::vectorise(U).t();
    int m = U0.n_cols; // dimension of solution space
    if (m == 0) { // no initial conditions err
        std::cerr << "am1() failed: no initial condition input." << std::endl;
        return;
    }

    double t0 = t(0);
    double tf = t(1);
    t = arma::zeros(20);
    t(0) = t0;
    U = arma::zeros(20,m); 
    U.row(0) = U0;

    broyd fsolver;
    fsolver.tol = max_nonlin_err;
    fsolver.max_iterations = max_nonlin_iter;

    unsigned long long i = 1;
    while (t(i) <= tf) {
        t(i) = t(i-1) + k;
        if (t(i) > tf) {
            t(i) = tf;
            k = tf - t(i-1);
        }

        arma::vec UU = U.row(i-1).t();
        auto eulerStep = [&f,&U,&t,i,k](const arma::vec& x) -> arma::vec {
            arma::rowvec z = x.t() - (  U.row(i-1) + k * f(t(i), x.t())  );
            return z.t();
        };
        fsolver.fsolve(eulerStep, UU);

        double kk = event_handle(t(i-1), U.row(i-1), t(i), UU.t(), k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        U.row(i) = UU.t();
        if (t(i) == tf) break; // t span stop
        if (kk == 0) break; // event stop
        
        i++;                        // move to the next step
        if (i+1 == t.n_rows) {
            t.resize(t.n_rows*2,1);
            U.resize(U.n_rows*2,U.n_cols);
        }
    }
    t = t( arma::span(0,i) );
    U = U.rows( arma::span(0,i) );
}

/* ode_solve(f, jacobian, t, U) : multivariate implicit Euler method O(k) for any explicit first order system of ODEs.
 * our equations are of the form u' = f(t,u) [u must be a row vector].
 * --- f  : f(t,u) [t must be the first variable, u the second].
 * --- jacobian : J(t,u) jacobian matrix of f(t,u) with respect to u.
 * --- t  : vector to store t-values initialized at {t_initial, t_final}.
 * --- U  : vector to store the solution first row must be u(t0). */
void numerics::ode::am1::ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                                   const std::function<arma::mat(double, const arma::vec&)>& jacobian,
                                   arma::vec& t, arma::mat& U) {
    double k = step;

    arma::rowvec U0 = arma::vectorise(U).t();
    int m = U0.n_cols; // dimension of solution space
    if (m == 0) { // no initial conditions err
        std::cerr << "am1() failed: no initial condition input." << std::endl;
        return;
    }

    double t0 = t(0);
    double tf = t(1);
    t = arma::zeros(20);
    t(0) = t0;
    U = arma::zeros(20,m); 
    U.row(0) = U0;

    newton fsolver;
    fsolver.tol = max_nonlin_err;
    fsolver.max_iterations = max_nonlin_iter;

    unsigned long long i = 1;
    while (t(i) <= tf) {
        t(i) = t(i-1) + k;
        if (t(i) > tf) {
            t(i) = tf;
            k = tf - t(i-1);
        }

        arma::vec UU = U.row(i-1).t();
        auto eulerStep = [&f,&U,&t,i,k](const arma::vec& x) -> arma::vec {
            arma::rowvec z = x.t() - (  U.row(i-1) + k * f(t(i), x.t())  );
            return z.t();
        };
        auto euler_jac = [&jacobian,&U,&t,i,k](const arma::vec& x) -> arma::mat {
            arma::mat J = jacobian(t(i), x.t());
            J = arma::eye(arma::size(J)) - k*J;
            return J;
        };
        fsolver.fsolve(eulerStep, euler_jac, UU);

        double kk = event_handle(t(i-1), U.row(i-1), t(i), UU.t(), k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        U.row(i) = UU.t();
        if (t(i) == tf) break; // t span stop
        if (kk == 0) break; // event stop
        
        i++;                        // move to the next step
        if (i+1 == t.n_rows) {
            t.resize(t.n_rows*2,1);
            U.resize(U.n_rows*2,U.n_cols);
        }
    }
    t = t( arma::span(0,i) );
    U = U.rows( arma::span(0,i) );
}