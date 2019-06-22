#include <numerics.hpp>

/* ode_solve(f, t, U) : multivariate Adams-Multon O(k^2) method for any explicit first order system of ODEs.
 * our equations are of the form u' = f(t,u) [u must be a row vector].
 * --- f  : f(t,u) [t must be the first variable, u the second].
 * --- t  : vector to store t-values initialized at {t_initial, t_final}.
 * --- U  : vector to store the solution first row must be u(t0). */
void numerics::ode::am2::ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U) {
    double k = step;
    
    arma::rowvec U0 = arma::vectorise(U).t();
    int m = U0.n_cols; // dimension of solution space
    if (m == 0) {
        std::cerr << "am2() failed: no initial condition input." << std::endl;
        return;
    }

    double t0 = t(0);
    double tf = t(1);
    t = arma::zeros(20);
    t(0) = t0;
    t(1) = t0 + k;

    U = arma::zeros(20,m);
    U.row(0) = U0;

    broyd fsolver;
    fsolver.tol = max_nonlin_err;
    fsolver.max_iterations = max_nonlin_iter;

    arma::vec V = U0.t();
    auto backEulerStep = [&f,&U0,&t,k](const arma::vec& x) -> arma::vec {
        arma::rowvec z = U0 + k * f(t(1),x.t());
        return x - z.t();
    };
    fsolver.fsolve(backEulerStep,V);
    U.row(1) = V.t();

    unsigned short i = 2;
    while (t(i) <= tf) {
        t(i) = t(i-1) + k;
        if (t(i) > tf) {
            t(i) = tf;
            k = tf - t(i);
        }

        V = U.row(i-1).t();
        auto trapStep = [&f,&U,&t,i,k](const arma::vec& x) -> arma::vec {
            arma::rowvec z = U.row(i-1) + (k/2) * ( f(t(i-1),U.row(i-1)) + f(t(i), x.t()) );
            return x - z.t();
        };
        fsolver.fsolve(backEulerStep,V);

        double kk = event_handle(t(i-1), U.row(i-1), t(i), V.t(), k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        U.row(i) = V.t();
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

/* ode_solve(f, jacobian, t, U) : multivariate Adams-Multon O(k^2) method for any explicit first order system of ODEs.
 * our equations are of the form u' = f(t,u) [u must be a row vector].
 * --- f  : f(t,u) [t must be the first variable, u the second].
 * --- jacobian : J(t,u) jacobian matrix of f(t,u) with repect to u.
 * --- t  : vector to store t-values initialized at {t_initial, t_final}.
 * --- U  : vector to store the solution first row must be u(t0). */
void numerics::ode::am2::ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                                   const std::function<arma::mat(double, const arma::rowvec&)>& jacobian,
                                   arma::vec& t, arma::mat& U) {
    double k = step;
    
    arma::rowvec U0 = arma::vectorise(U).t();
    int m = U0.n_cols; // dimension of solution space
    if (m == 0) {
        std::cerr << "am2() failed: no initial condition input." << std::endl;
        return;
    }

    double t0 = t(0);
    double tf = t(1);
    t = arma::zeros(20);
    t(0) = t0;
    t(1) = t0 + k;

    U = arma::zeros(20,m);
    U.row(0) = U0;

    newton fsolver;
    fsolver.tol = max_nonlin_err;
    fsolver.max_iterations = max_nonlin_iter;

    arma::vec V = U0.t();
    auto backEulerStep = [&f,&U0,&t,k](const arma::vec& x) -> arma::vec {
        arma::rowvec z = U0 + k * f(t(1),x.t());
        return x - z.t();
    };
    auto euler_jac = [&jacobian,&U0,&t,k](const arma::vec& x) -> arma::mat {
        arma::mat J = jacobian(t(1),x.t());
        J = arma::eye(arma::size(J)) - k*J;
        return J;
    };
    fsolver.fsolve(backEulerStep, euler_jac, V);
    U.row(1) = V.t();

    unsigned short i = 2;
    while (t(i) <= tf) {
        t(i) = t(i-1) + k;
        if (t(i) > tf) {
            t(i) = tf;
            k = tf - t(i);
        }

        V = U.row(i-1).t();
        auto trapStep = [&f,&U,&t,i,k](const arma::vec& x) -> arma::vec {
            arma::rowvec z = U.row(i-1) + (k/2) * ( f(t(i-1),U.row(i-1)) + f(t(i), x.t()) );
            return x - z.t();
        };
        auto trap_jac = [&jacobian,&U,&t,i,k](const arma::vec& x) -> arma::mat {
            arma::mat J = jacobian(t(i), x.t());
            J = arma::eye(arma::size(J)) - (k/2) * J;
            return J;
        };
        fsolver.fsolve(backEulerStep, trap_jac, V);

        double kk = event_handle(t(i-1), U.row(i-1), t(i), V.t(), k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        U.row(i) = V.t();
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