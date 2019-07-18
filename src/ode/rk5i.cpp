#include <numerics.hpp>

/* ode_solve(f, t, U) : diagonally implicit runge kutta O(K^5) method for any explicit first order system of ODEs.
 * our equations are of the form u' = f(t,u) [u must be a row vector].
 * --- f  : f(t,u) [t must be the first variable, u the second].
 * --- t  : vector to store t-values initialized at {t_initial, t_final}.
 * --- U  : vector to store the solution first row must be u(t0). */
void numerics::ode::rk5i::ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U) {
    double k = step;
    arma::rowvec U0 = arma::vectorise(U).t();
    int m = U0.n_cols; // dimension of solution space
    if (m == 0) { // no initial conditions err
        std::cerr << "rk5i() failed: no initial condition input." << std::endl;
        return;
    }

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
        V1 = f(t(i-1), U.row(i-1));

        arma::vec V_temp = U.row(i-1).t();
        auto v2f = [&f,&U,&t,&V1,i,k](const arma::vec& x) -> arma::vec {
            arma::rowvec z = f(t(i-1), U.row(i-1) + 0.125*k*V1 + 0.125*k*x.t());
            return x - z.t();
        };
        fsolver.fsolve(v2f,V_temp);

        V2 = V_temp.t();
        auto v3f = [&f,&U,&t,&V1,&V2,i,k](const arma::vec& x) -> arma::vec {
            arma::rowvec z = f(t(i-1) + 0.7*k, U.row(i-1) - 0.01*k*V1 + 0.56*k*V2 + 0.15*k*x.t());
            return x - z.t();
        };
        fsolver.fsolve(v3f,V_temp);
        V3 = V_temp.t();
    
        V4 = f(t(i-1) + k, U.row(i-1) + (2.0/7)*k*V1 + (5.0/7)*k*V3);

        arma::rowvec rk5 = U.row(i-1) + k * ( (1.0/14)*V1 + (32.0/81)*V2 + (250.0/567)*V3 + (5.0/54)*V4 );
        
        double kk = event_handle(t(i-1), U.row(i-1), t(i), rk5, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        U.row(i) = rk5;
        if (t(i) >= tf) break; // t span stop
        if (kk == 0) break; // event stop
        i++;                        // move to the next step
        if (i+1 >= t.n_rows) {
            t.resize(t.n_rows*2,1);
            U.resize(U.n_rows*2,U.n_cols);
        }
    }
    t = t( arma::span(0,i-1) );
    U = U.rows( arma::span(0,i-1) );
}

/* ode_solve(f, jacobian, t, U) : diagonally implicit runge kutta O(K^5) method for any explicit first order system of ODEs.
 * our equations are of the form u' = f(t,u) [u must be a row vector].
 * --- f  : f(t,u) [t must be the first variable, u the second].
 * --- jacobian : J(t,u) jacobian of f(t,u) with respect to u.
 * --- t  : vector to store t-values initialized at {t_initial, t_final}.
 * --- U  : vector to store the solution first row must be u(t0). */
void numerics::ode::rk5i::ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                                    const std::function<arma::mat(double,const arma::rowvec&)>& jacobian,
                                    arma::vec& t, arma::mat& U) {
    double k = step;
    arma::rowvec U0 = arma::vectorise(U).t();
    int m = U0.n_cols; // dimension of solution space
    if (m == 0) { // no initial conditions err
        std::cerr << "rk5i() failed: no initial condition input." << std::endl;
        return;
    }

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
        V1 = f(t(i-1), U.row(i-1));

        arma::vec V_temp = U.row(i-1).t();
        auto v2f = [&f,&U,&t,&V1,i,k](const arma::vec& x) -> arma::vec {
            arma::rowvec z = f(t(i-1), U.row(i-1) + 0.125*k*V1 + 0.125*k*x.t());
            return x - z.t();
        };
        auto v2f_jac = [&jacobian,&U,&t,&V1,i,k](const arma::vec& x) -> arma::mat {
            arma::mat J = jacobian(t(i-1), U.row(i-1) + 0.125*k*V1 + 0.125*k*x.t());
            J = arma::eye(arma::size(J)) - 0.125*k*J;
            return J;
        };
        fsolver.fsolve(v2f,v2f_jac,V_temp);

        V2 = V_temp.t();
        auto v3f = [&f,&U,&t,&V1,&V2,i,k](const arma::vec& x) -> arma::vec {
            arma::rowvec z = f(t(i-1) + 0.7*k, U.row(i-1) - 0.01*k*V1 + 0.56*k*V2 + 0.15*k*x.t());
            return x - z.t();
        };
        auto v3f_jac = [&jacobian,&U,&t,&V1,&V2,i,k](const arma::vec& x) -> arma::mat {
            arma::mat J = jacobian(t(i-1) + 0.7*k, U.row(i-1) - 0.01*k*V1 + 0.56*k*V2 + 0.15*k*x.t());
            J = arma::eye(arma::size(J)) - 0.15*k*J;
            return J;
        };
        fsolver.fsolve(v3f,v3f_jac,V_temp);
        V3 = V_temp.t();
    
        V4 = f(t(i-1) + k, U.row(i-1) + (2.0/7)*k*V1 + (5.0/7)*k*V3);

        arma::rowvec rk5 = U.row(i-1) + k * ( (1.0/14)*V1 + (32.0/81)*V2 + (250.0/567)*V3 + (5.0/54)*V4 );
        
        double kk = event_handle(t(i-1), U.row(i-1), t(i), rk5, k);
        if (0 < kk && kk < k) {
            k = kk;
            continue;
        }

        U.row(i) = rk5;
        if (t(i) >= tf) break; // t span stop
        if (kk == 0) break; // event stop
        i++;                        // move to the next step
        if (i+1 >= t.n_rows) {
            t.resize(t.n_rows*2,1);
            U.resize(U.n_rows*2,U.n_cols);
        }
    }
    t = t( arma::span(0,i) );
    U = U.rows( arma::span(0,i) );
}