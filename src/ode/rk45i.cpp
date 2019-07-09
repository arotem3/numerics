#include <numerics.hpp>

/* ode_solve(f, t, U) : adaptive diagonally implicit runge kutta O(K^4) method for any explicit first order system of ODEs.
 * our equations are of the form u' = f(t,u) [u must be a row vector].
 * --- f  : f(t,u) [t must be the first variable, u the second].
 * --- t  : vector to store t-values initialized at {t_initial, t_final}.
 * --- U  : vector to store the solution first row must be u(t0). */
void numerics::ode::rk45i::ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U) {
    arma::rowvec U0 = arma::vectorise(U).t();
    int m = U0.n_cols; // dimension of solution space
    if (m == 0) { // no initial conditions err
        std::cerr << "rk45i() failed: no initial condition input." << std::endl;
        return;
    }

    double t0 = t(0);
    double tf = t(1);
    t = arma::zeros(20);
    t(0) = t0;

    U = arma::zeros(20,m);
    U.row(0) = U0;
    double k = std::max(0.01, (tf - t0)/100);

    broyd fsolver;
    fsolver.tol = max_nonlin_err;
    fsolver.max_iterations = max_nonlin_iter;

    arma::vec v1,v2,v3,v4,v5,z;
    arma::rowvec u4,u5;
    unsigned short i = 0;
    while (t(i) <= tf) {
        v1 = k*f(t(i), U.row(i)).t();
        fsolver.fsolve(
            [k,i,&f,&t,&U](const arma::vec& u) -> arma::vec {
                arma::rowvec r = k*f(t(i) + k/4, U.row(i) + u.t()/4);
                return r.t() - u;
            }, v1
        );

        z = U.row(i).t() + v1/2;
        v2 = z;
        fsolver.fsolve(
            [k,i,&f,&t,&U,&z](const arma::vec& u) -> arma::vec {
                arma::rowvec r = k*f(t(i) + 3*k/4, z.t() + u.t()/4);
                return r.t() - u;
            }, v2
        );

        z = U.row(i).t() + 17*v1/50 - v2/25;
        v3 = z;
        fsolver.fsolve(
            [k,i,&f,&t,&U,&z](const arma::vec& u) -> arma::vec {
                arma::rowvec r = k*f(t(i) + 11*k/20, z.t() + u.t()/4);
                return r.t() - u;
            }, v3
        );

        z = U.row(i).t() + 371*v1/1360 - 137*v2/2720 + 15*v3/544;
        v4 = z;
        fsolver.fsolve(
            [k,i,&f,&t,&U,&z](const arma::vec& u) -> arma::vec {
                arma::rowvec r = k*f(t(i) + k/2, z.t() + u.t()/4);
                return r.t() - u;
            }, v4
        );

        z = U.row(i).t() + 25*v1/24 - 49*v2/48 + 125*v3/16 - 85*v4/12;
        arma::mat J = k*approx_jacobian(
            [k,i,&f,&t](const arma::vec& u) -> arma::vec {
                return f(t(i) + k, u.t()).t();
            }, z, k*k
        );
        v5 = arma::solve( arma::eye(arma::size(J)) - J/4, k*f(t(i) + k, z.t()).t() );
        u4 = (z + v5/4).t();
        
        v5 = z;
        fsolver.fsolve(
            [k,i,&f,&t,&U,&z](const arma::vec& u)->arma::vec {
                arma::rowvec r = k*f(t(i) + k, z.t() + u.t()/4);
                return r.t() - u;
            }, v5
        );

        u5 = (z + v5/4).t();
        double err = arma::norm(u4 - u5, "inf");

        double kk = 2*k;
        if (i > 0) kk = event_handle(t(i), U.row(i), t(i) + k,u5,k);

        if (err < adaptive_max_err) {
            if (0 <  kk && kk < k) {
                k = kk;
                continue;
            }

            t(i+1) = t(i) + k;
            U.row(i+1) = u5;
            i++;
            
            if (i+1 >= t.n_rows) {
                t.resize(t.n_rows*2,1);
                U.resize(U.n_rows*2,U.n_cols);
            }
        }

        if (kk == 0) break;
        k *= std::min(6.0, std::max(0.2, 0.9*std::pow(adaptive_max_err/err,0.2)));
        if (t(i) + k > tf) k = tf - t(i);
        if (k < adaptive_step_min) {
            std::cerr << "rk45i() failed: method does not converge b/c minimum k exceeded." << std::endl;
            std::cerr << "\tfailed at t = " << t(i) << std::endl;
            break;
        } else if (k > adaptive_step_max) k = adaptive_step_max;
    }
    t = t( arma::span(0,i) );
    U = U.rows( arma::span(0,i) );
}

/* ode_solve(f, jacobian, t, U) : adaptive diagonally implicit runge kutta O(K^4) method for any explicit first order system of ODEs.
 * our equations are of the form u' = f(t,u) [u must be a row vector].
 * --- f  : f(t,u) [t must be the first variable, u the second].
 * --- jacobian : J(t,u) jacobian of f(t,u) with respect to u.
 * --- t  : vector to store t-values initialized at {t_initial, t_final}.
 * --- U  : vector to store the solution first row must be u(t0). */
void numerics::ode::rk45i::ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                                    const std::function<arma::mat(double,const arma::rowvec&)>& jacobian,
                                    arma::vec& t, arma::mat& U) {
    arma::rowvec U0 = arma::vectorise(U).t();
    int m = U0.n_cols; // dimension of solution space
    if (m == 0) { // no initial conditions err
        std::cerr << "rk45i() failed: no initial condition input." << std::endl;
        return;
    }

    double t0 = t(0);
    double tf = t(1);
    t = arma::zeros(20);
    t(0) = t0;

    U = arma::zeros(20,m);
    U.row(0) = U0;
    double k = std::max(0.01, (tf - t0)/100);

    broyd fsolver;
    fsolver.tol = max_nonlin_err;
    fsolver.max_iterations = max_nonlin_iter;

    arma::vec v1,v2,v3,v4,v5,z;
    arma::rowvec u4,u5;
    unsigned short i = 0;
    while (t(i) <= tf) {
        v1 = k*f(t(i), U.row(i)).t();
        fsolver.fsolve(
            [k,i,&f,&t,&U](const arma::vec& u) -> arma::vec {
                arma::rowvec r = k*f(t(i) + k/4, U.row(i) + u.t()/4);
                return r.t() - u;
            },
            [k,i,&f,&jacobian,&t,&U](const arma::vec& u) -> arma::mat {
                arma::mat J = jacobian(t(i) + k/4, U.row(i) + u.t()/4);
                return arma::eye(arma::size(J)) - k/4*J;
            }, v1
        );

        z = U.row(i).t() + v1/2;
        v2 = z;
        fsolver.fsolve(
            [k,i,&f,&t,&U,&z](const arma::vec& u) -> arma::vec {
                arma::rowvec r = k*f(t(i) + 3*k/4, z.t() + u.t()/4);
                return r.t() - u;
            },
            [k,i,&f,&jacobian,&t,&U,&z](const arma::vec& u) -> arma::mat {
                arma::mat J = jacobian(t(i) + 3*k/4, z.t() + u.t()/4);
                return arma::eye(arma::size(J)) - k/4*J;
            }, v2
        );

        z = U.row(i).t() + 17*v1/50 - v2/25;
        v3 = z;
        fsolver.fsolve(
            [k,i,&f,&t,&U,&z](const arma::vec& u) -> arma::vec {
                arma::rowvec r = k*f(t(i) + 11*k/20, z.t() + u.t()/4);
                return r.t() - u;
            },
            [k,i,&f,&jacobian,&t,&U,&z](const arma::vec& u) -> arma::mat {
                arma::mat J = jacobian(t(i) + 11*k/20, z.t() + u.t()/4);
                return arma::eye(arma::size(J)) - k/4*J;
            }, v3
        );

        z = U.row(i).t() + 371*v1/1360 - 137*v2/2720 + 15*v3/544;
        v4 = z;
        fsolver.fsolve(
            [k,i,&f,&t,&U,&z](const arma::vec& u) -> arma::vec {
                arma::rowvec r = k*f(t(i) + k/2, z.t() + u.t()/4);
                return r.t() - u;
            },
            [k,i,&f,&jacobian,&t,&U,&z](const arma::vec& u) -> arma::mat {
                arma::mat J = jacobian(t(i) + k/2, z.t() + u.t()/4);
                return arma::eye(arma::size(J)) - k/4*J;
            }, v4
        );

        z = U.row(i).t() + 25*v1/24 - 49*v2/48 + 125*v3/16 - 85*v4/12;
        arma::mat J = k*approx_jacobian(
            [k,i,&f,&t](const arma::vec& u) -> arma::vec {
                return f(t(i) + k, u.t()).t();
            }, z, k*k
        );
        v5 = arma::solve( arma::eye(arma::size(J)) - J/4, k*f(t(i) + k, z.t()).t() );
        u4 = (z + v5/4).t();
        
        v5 = z;
        fsolver.fsolve(
            [k,i,&f,&t,&U,&z](const arma::vec& u)->arma::vec {
                arma::rowvec r = k*f(t(i) + k, z.t() + u.t()/4);
                return r.t() - u;
            },
            [k,i,&f,&jacobian,&t,&U,&z](const arma::vec& u) -> arma::mat {
                arma::mat J = jacobian(t(i) + k, z.t() + u.t()/4);
                return arma::eye(arma::size(J)) - k/4*J;
            }, v5
        );

        u5 = (z + v5/4).t();
        double err = arma::norm(u4 - u5, "inf");

        double kk = 2*k;
        if (i > 0) kk = event_handle(t(i), U.row(i), t(i) + k,u5,k);

        if (err < adaptive_max_err) {
            if (0 <  kk && kk < k) {
                k = kk;
                continue;
            }

            t(i+1) = t(i) + k;
            U.row(i+1) = u5;
            i++;
            
            if (i+1 >= t.n_rows) {
                t.resize(t.n_rows*2,1);
                U.resize(U.n_rows*2,U.n_cols);
            }
        }

        if (kk == 0) break;
        k *= std::min(6.0, std::max(0.2, 0.9*std::pow(adaptive_max_err/err,0.2)));
        if (t(i) + k > tf) k = tf - t(i);
        if (k < adaptive_step_min) {
            std::cerr << "rk45i() failed: method does not converge b/c minimum k exceeded." << std::endl;
            std::cerr << "\tfailed at t = " << t(i) << std::endl;
            break;
        } else if (k > adaptive_step_max) k = adaptive_step_max;
    }
    t = t( arma::span(0,i) );
    U = U.rows( arma::span(0,i) );
}