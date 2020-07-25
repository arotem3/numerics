#include <numerics.hpp>

numerics::ode::ODESolution numerics::ode::rk45i::ode_solve(const odefunc& f, double t0, double tf, const arma::vec& U0) {
    _check_range(t0, tf);
    double k = (tf - t0) / 100;

    u_long m = U0.n_elem;
    ODESolution sol(m);

    std::vector<double>& t = sol._tvec;
    std::vector<arma::vec>& U = sol._Uvec;
    t.push_back(t0);
    U.push_back(U0);

    optimization::Broyd fsolver(_max_solver_err, _max_solver_iter);

    arma::vec v1,v2,v3,v4,v5,z;
    arma::vec u4,u5;
    unsigned long long i = 0;
    while (t.at(i) < tf) {
        v1 = k*f(t.at(i), U.at(i));
        fsolver.fsolve(
            v1,
            [k,i,&f,&t,&U](const arma::vec& u) -> arma::vec {
                arma::vec r = k*f(t.at(i) + k/4, U.at(i) + u/4);
                return r - u;
            }
        );

        z = U.at(i) + v1/2;
        v2 = z;
        fsolver.fsolve(
            v2,
            [k,i,&f,&t,&U,&z](const arma::vec& u) -> arma::vec {
                arma::vec r = k*f(t.at(i) + 3*k/4, z + u/4);
                return r - u;
            }
        );

        z = U.at(i) + 17*v1/50 - v2/25;
        v3 = z;
        fsolver.fsolve(
            v3,
            [k,i,&f,&t,&U,&z](const arma::vec& u) -> arma::vec {
                arma::vec r = k*f(t.at(i) + 11*k/20, z + u/4);
                return r - u;
            }
        );

        z = U.at(i) + 371*v1/1360 - 137*v2/2720 + 15*v3/544;
        v4 = z;
        fsolver.fsolve(
            v4,
            [k,i,&f,&t,&U,&z](const arma::vec& u) -> arma::vec {
                arma::vec r = k*f(t.at(i) + k/2, z + u/4);
                return r - u;
            }
        );

        z = U.at(i) + 25*v1/24 - 49*v2/48 + 125*v3/16 - 85*v4/12;
        v5 = z;
        fsolver.fsolve(
            v5,
            [k,i,&f,&t,&U,&z](const arma::vec& u)->arma::vec {
                arma::vec r = k*f(t.at(i) + k, z + u/4);
                return r - u;
            }
        );

        u4 = U.at(i) + (59*v1/48 - 17*v2/96 + 225*v3/32 - 85*v4/12);
        u5 = (z + v5/4);
        double err = arma::norm(u4 - u5, "inf");

        double kk;
        if (i > 0) kk = event_handle(t.at(i), U.at(i), t.at(i) + k,u5,k);
        else kk = 2*k; // dummy initialization to ensure kk > k for the first iter

        if (err < _max_err*arma::norm(U.at(i),"inf")) {
            if (0 <  kk && kk < k) {
                k = kk;
                continue;
            }

            t.push_back(t.at(i) + k);
            U.push_back(u5);
            i++;
        }

        if (kk == 0) break;
        k *= std::min(10.0, std::max(0.1, 0.9*std::pow(_max_err/err,0.25)));
        if (k < _step_min) {
            std::cerr << "rk45i failed: method could not converge b/c current step-size (=" << k << ") < minimum step size (=" << _step_min << ")\n";
            std::cerr << "\tfailed at t = " << t.at(i) << "\n";
            break;
        }
        if (t.at(i) + k > tf) k = tf - t.at(i);
    }
    sol._prepare();
    return sol;
}

numerics::ode::ODESolution numerics::ode::rk45i::ode_solve(const odefunc& f, const odejacobian& jacobian, double t0, double tf, const arma::vec& U0) {
    _check_range(t0, tf);
    double k = (tf - t0) / 100;

    u_long m = U0.n_elem;
    ODESolution sol(m);

    std::vector<double>& t = sol._tvec;
    std::vector<arma::vec>& U = sol._Uvec;
    t.push_back(t0);
    U.push_back(U0);

    optimization::Newton fsolver(_max_solver_err, _max_solver_iter);

    arma::vec v1,v2,v3,v4,v5,z;
    arma::vec u4,u5;
    unsigned long long i = 0;
    while (t.at(i) < tf) {
        v1 = k*f(t.at(i), U.at(i));
        fsolver.fsolve(
            v1,
            [k,i,&f,&t,&U](const arma::vec& u) -> arma::vec {
                arma::vec r = k*f(t.at(i) + k/4, U.at(i) + u/4);
                return r - u;
            },
            [k,i,&f,&jacobian,&t,&U](const arma::vec& u) -> arma::mat {
                arma::mat J = jacobian(t.at(i) + k/4, U.at(i) + u/4);
                return arma::eye(arma::size(J)) - k/4*J;
            }
        );

        z = U.at(i) + v1/2;
        v2 = z;
        fsolver.fsolve(
            v2,
            [k,i,&f,&t,&U,&z](const arma::vec& u) -> arma::vec {
                arma::vec r = k*f(t.at(i) + 3*k/4, z + u/4);
                return r - u;
            },
            [k,i,&f,&jacobian,&t,&U,&z](const arma::vec& u) -> arma::mat {
                arma::mat J = jacobian(t.at(i) + 3*k/4, z + u/4);
                return arma::eye(arma::size(J)) - k/4*J;
            }
        );

        z = U.at(i) + 17*v1/50 - v2/25;
        v3 = z;
        fsolver.fsolve(
            v3,
            [k,i,&f,&t,&U,&z](const arma::vec& u) -> arma::vec {
                arma::vec r = k*f(t.at(i) + 11*k/20, z + u/4);
                return r - u;
            },
            [k,i,&f,&jacobian,&t,&U,&z](const arma::vec& u) -> arma::mat {
                arma::mat J = jacobian(t.at(i) + 11*k/20, z + u/4);
                return arma::eye(arma::size(J)) - k/4*J;
            }
        );

        z = U.at(i) + 371*v1/1360 - 137*v2/2720 + 15*v3/544;
        v4 = z;
        fsolver.fsolve(
            v4,
            [k,i,&f,&t,&U,&z](const arma::vec& u) -> arma::vec {
                arma::vec r = k*f(t.at(i) + k/2, z + u/4);
                return r - u;
            },
            [k,i,&f,&jacobian,&t,&U,&z](const arma::vec& u) -> arma::mat {
                arma::mat J = jacobian(t.at(i) + k/2, z + u/4);
                return arma::eye(arma::size(J)) - k/4*J;
            }
        );

        z = U.at(i) + 25*v1/24 - 49*v2/48 + 125*v3/16 - 85*v4/12;
        v5 = z;
        fsolver.fsolve(
            v5,
            [k,i,&f,&t,&U,&z](const arma::vec& u)->arma::vec {
                arma::vec r = k*f(t.at(i) + k, z + u/4);
                return r - u;
            },
            [k,i,&f,&jacobian,&t,&U,&z](const arma::vec& u) -> arma::mat {
                arma::mat J = jacobian(t.at(i) + k, z + u/4);
                return arma::eye(arma::size(J)) - k/4*J;
            }
        );

        u4 = U.at(i) + (59*v1/48 - 17*v2/96 + 225*v3/32 - 85*v4/12);
        u5 = (z + v5/4);
        double err = arma::norm(u4 - u5, "inf");

        double kk;
        if (i > 0) kk = event_handle(t.at(i), U.at(i), t.at(i) + k,u5,k);
        else kk = 2*k; // dummy initialization to ensure kk > k for the first iter

        if (err < _max_err*arma::norm(U.at(i),"inf")) {
            if (0 <  kk && kk < k) {
                k = kk;
                continue;
            }

            t.push_back(t.at(i) + k);
            U.push_back(u5);
            i++;
        }

        if (kk == 0) break;
        k *= std::min(10.0, std::max(0.1, 0.9*std::pow(_max_err/err,0.25)));
        if (k < _step_min) {
            std::cerr << "rk45i failed: method could not converge b/c current step-size (=" << k << ") < minimum step size (=" << _step_min << ")\n";
            std::cerr << "\tfailed at t = " << t.at(i) << "\n";
            break;
        }
        if (t.at(i) + k > tf) k = tf - t.at(i);
    }
    sol._prepare();
    return sol;
}