#include <numerics.hpp>

void numerics::ode::rk45i::solve_ivp(const odefunc& f, double t0, double tf, const arma::vec& U0) {
    _check_range(t0, tf);
    double k = (tf - t0) / 100;

    _t.clear(); _U.clear();
    _t.push_back(t0);
    _U.push_back(U0);

    optimization::Broyd fsolver(_max_solver_err, _max_solver_iter);

    arma::vec v1,v2,v3,v4,v5,z;
    arma::vec u4,u5;
    unsigned long long i = 0;
    while (_t.at(i) < tf) {
        v1 = k*f(_t.at(i), _U.at(i));
        fsolver.fsolve(
            v1,
            [this,k,i,&f](const arma::vec& u) -> arma::vec {
                arma::vec r = k*f(_t.at(i) + k/4, _U.at(i) + u/4);
                return r - u;
            }
        );

        z = _U.at(i) + v1/2;
        v2 = z;
        fsolver.fsolve(
            v2,
            [this,k,i,&f,&z](const arma::vec& u) -> arma::vec {
                arma::vec r = k*f(_t.at(i) + 3*k/4, z + u/4);
                return r - u;
            }
        );

        z = _U.at(i) + 17*v1/50 - v2/25;
        v3 = z;
        fsolver.fsolve(
            v3,
            [this,k,i,&f,&z](const arma::vec& u) -> arma::vec {
                arma::vec r = k*f(_t.at(i) + 11*k/20, z + u/4);
                return r - u;
            }
        );

        z = _U.at(i) + 371*v1/1360 - 137*v2/2720 + 15*v3/544;
        v4 = z;
        fsolver.fsolve(
            v4,
            [this,k,i,&f,&z](const arma::vec& u) -> arma::vec {
                arma::vec r = k*f(_t.at(i) + k/2, z + u/4);
                return r - u;
            }
        );

        z = _U.at(i) + 25*v1/24 - 49*v2/48 + 125*v3/16 - 85*v4/12;
        v5 = z;
        fsolver.fsolve(
            v5,
            [this,k,i,&f,&z](const arma::vec& u)->arma::vec {
                arma::vec r = k*f(_t.at(i) + k, z + u/4);
                return r - u;
            }
        );

        u4 = _U.at(i) + (59*v1/48 - 17*v2/96 + 225*v3/32 - 85*v4/12);
        u5 = (z + v5/4);
        double err = arma::norm(u4 - u5, "inf");

        double kk;
        if (i > 0) kk = event_handle(_t.at(i), _U.at(i), _t.at(i) + k,u5,k);
        else kk = 2*k; // dummy initialization to ensure kk > k for the first iter

        if (err < _max_err*arma::norm(_U.at(i),"inf")) {
            if (0 <  kk && kk < k) {
                k = kk;
                continue;
            }

            _t.push_back(_t.at(i) + k);
            _U.push_back(u5);
            i++;
        }

        if (kk == 0) break;
        k *= std::min(10.0, std::max(0.1, 0.9*std::pow(_max_err/err,0.25)));
        if (k < _step_min) {
            std::cerr << "rk45i failed: method could not converge b/c current step-size (=" << k << ") < minimum step size (=" << _step_min << ")\n";
            std::cerr << "\tfailed at _t = " << _t.at(i) << "\n";
            break;
        }
        if (_t.at(i) + k > tf) k = tf - _t.at(i);
    }
}

void numerics::ode::rk45i::solve_ivp(const odefunc& f, const odejacobian& jacobian, double t0, double tf, const arma::vec& U0) {
    _check_range(t0, tf);
    double k = (tf - t0) / 100;

    _t.clear(); _U.clear();
    _t.push_back(t0);
    _U.push_back(U0);

    optimization::Newton fsolver(_max_solver_err, _max_solver_iter);

    arma::vec v1,v2,v3,v4,v5,z;
    arma::vec u4,u5;
    unsigned long long i = 0;
    while (_t.at(i) < tf) {
        v1 = k*f(_t.at(i), _U.at(i));
        fsolver.fsolve(
            v1,
            [this,k,i,&f](const arma::vec& u) -> arma::vec {
                arma::vec r = k*f(_t.at(i) + k/4, _U.at(i) + u/4);
                return r - u;
            },
            [this,k,i,&f,&jacobian](const arma::vec& u) -> arma::mat {
                arma::mat J = jacobian(_t.at(i) + k/4, _U.at(i) + u/4);
                return arma::eye(arma::size(J)) - k/4*J;
            }
        );

        z = _U.at(i) + v1/2;
        v2 = z;
        fsolver.fsolve(
            v2,
            [this,k,i,&f,&z](const arma::vec& u) -> arma::vec {
                arma::vec r = k*f(_t.at(i) + 3*k/4, z + u/4);
                return r - u;
            },
            [this,k,i,&f,&jacobian,&z](const arma::vec& u) -> arma::mat {
                arma::mat J = jacobian(_t.at(i) + 3*k/4, z + u/4);
                return arma::eye(arma::size(J)) - k/4*J;
            }
        );

        z = _U.at(i) + 17*v1/50 - v2/25;
        v3 = z;
        fsolver.fsolve(
            v3,
            [this,k,i,&f,&z](const arma::vec& u) -> arma::vec {
                arma::vec r = k*f(_t.at(i) + 11*k/20, z + u/4);
                return r - u;
            },
            [this,k,i,&f,&jacobian,&z](const arma::vec& u) -> arma::mat {
                arma::mat J = jacobian(_t.at(i) + 11*k/20, z + u/4);
                return arma::eye(arma::size(J)) - k/4*J;
            }
        );

        z = _U.at(i) + 371*v1/1360 - 137*v2/2720 + 15*v3/544;
        v4 = z;
        fsolver.fsolve(
            v4,
            [this,k,i,&f,&z](const arma::vec& u) -> arma::vec {
                arma::vec r = k*f(_t.at(i) + k/2, z + u/4);
                return r - u;
            },
            [this,k,i,&f,&jacobian,&z](const arma::vec& u) -> arma::mat {
                arma::mat J = jacobian(_t.at(i) + k/2, z + u/4);
                return arma::eye(arma::size(J)) - k/4*J;
            }
        );

        z = _U.at(i) + 25*v1/24 - 49*v2/48 + 125*v3/16 - 85*v4/12;
        v5 = z;
        fsolver.fsolve(
            v5,
            [this,k,i,&f,&z](const arma::vec& u)->arma::vec {
                arma::vec r = k*f(_t.at(i) + k, z + u/4);
                return r - u;
            },
            [this,k,i,&f,&jacobian,&z](const arma::vec& u) -> arma::mat {
                arma::mat J = jacobian(_t.at(i) + k, z + u/4);
                return arma::eye(arma::size(J)) - k/4*J;
            }
        );

        u4 = _U.at(i) + (59*v1/48 - 17*v2/96 + 225*v3/32 - 85*v4/12);
        u5 = (z + v5/4);
        double err = arma::norm(u4 - u5, "inf");

        double kk;
        if (i > 0) kk = event_handle(_t.at(i), _U.at(i), _t.at(i) + k,u5,k);
        else kk = 2*k; // dummy initialization to ensure kk > k for the first iter

        if (err < _max_err*arma::norm(_U.at(i),"inf")) {
            if (0 <  kk && kk < k) {
                k = kk;
                continue;
            }

            _t.push_back(_t.at(i) + k);
            _U.push_back(u5);
            i++;
        }

        if (kk == 0) break;
        k *= std::min(10.0, std::max(0.1, 0.9*std::pow(_max_err/err,0.25)));
        if (k < _step_min) {
            std::cerr << "rk45i failed: method could not converge b/c current step-size (=" << k << ") < minimum step size (=" << _step_min << ")\n";
            std::cerr << "\tfailed at _t = " << _t.at(i) << "\n";
            break;
        }
        if (_t.at(i) + k > tf) k = tf - _t.at(i);
    }
}