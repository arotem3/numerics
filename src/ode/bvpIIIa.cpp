#include <numerics.hpp>

void numerics::ode::BVP3a::solve_bvp(const odefunc& f, const boundary_conditions& bc, const arma::vec& x0, const arma::mat& U0) {
    auto jacobian = [&](double x, const arma::vec& u) -> arma::mat {
        return numerics::approx_jacobian(
            [&](const arma::vec& v) -> arma::vec {
                return f(x,v);
            },
            u
        );
    };
    solve_bvp(f, jacobian, bc, x0, U0);
}


void numerics::ode::BVP3a::solve_bvp(const odefunc& f, const odejacobian& jacobian, const boundary_conditions& bc, const arma::vec& x0, const arma::mat& U0) {
    u_long n = x0.n_elem;
    u_long dim = _check_dim(x0, U0);

    arma::mat I = arma::eye(dim,dim);

    _check_x(x0);
    double L = x0(0);
    double R = x0(n-1);
    std::map<double, arma::vec> sol;
    std::map<double, arma::vec> ff;
    std::vector<double> x = arma::conv_to<std::vector<double>>::from(x0);
    for (int j=0; j < n; ++j) {
        if (U0.n_rows == n) sol[x0(j)] = U0.row(j).as_col();
        else sol[x0(j)] = U0.col(j);
    }

    u_long iter = 0;

    arma::sp_mat J;
    arma::vec r;


    while (true) {
        n = x.size();

        auto setup_equations = [&](bool compute_jacobian) -> void {
            // we evaluate residual r of Runge-Kutta scheme with respect to current approximation
            // and compute the jacobian if necessary
            r.set_size(dim*n);
            r(arma::span(0,dim-1)) = bc(sol.at(L), sol.at(R));
            
            if (compute_jacobian) {
                J.zeros(dim*n,dim*n);
                J(arma::span(0,dim-1), arma::span(0,dim-1)) = numerics::approx_jacobian(
                    [&](const arma::vec& v) -> arma::vec {
                        return bc(v, sol.at(R));
                    },
                    sol.at(L)
                );
                J(arma::span(0,dim-1), arma::span((n-1)*dim, n*dim-1)) = numerics::approx_jacobian(
                    [&](const arma::vec& v) -> arma::vec {
                        return bc(sol.at(L), v);
                    },
                    sol.at(R)
                );
            }

            ff[L] = f(L, sol.at(L));
            arma::mat dfdu0, dfdu1;

            if (compute_jacobian) dfdu1 = jacobian(L, sol[L]);

            for (int j=1; j < n; ++j) {
                double h = x.at(j) - x.at(j-1);

                ff[x.at(j)] = f(x.at(j), sol.at(x.at(j)));

                const arma::vec& u1 = sol.at(x.at(j));
                const arma::vec& u0 = sol.at(x.at(j-1));

                arma::vec u_half = (u0 + u1)/2 - 0.125*h*(ff[x.at(j)] - ff[x.at(j-1)]);
                arma::vec f_half = f(x.at(j-1) + h/2, u_half);

                r(arma::span(dim*j, dim*(j+1)-1)) = u1 - u0 - h/6 * (ff[x.at(j-1)] + 4*f_half + ff[x.at(j)]);

                if (compute_jacobian) {
                    dfdu0 = std::move(dfdu1);
                    dfdu1 = jacobian(x.at(j), u1);
                    arma::mat dfdu_half = jacobian(x.at(j-1)+h/2, u_half);

                    J(arma::span(dim*j,dim*(j+1)-1),arma::span(dim*(j-1), dim*j-1)) = -I - h/6 * dfdu0 - 2*h/3 * dfdu_half * (0.5*I + 0.125*h*dfdu0);
                    J(arma::span(dim*j,dim*(j+1)-1),arma::span(dim*j,dim*(j+1)-1)) = I - h/6 * dfdu1 - 2*h/3 * dfdu_half * (0.5*I - 0.125*h*dfdu1);
                }
            }
        };

        for (u_int ii=0; ii < 10; ++ii) { // upto ten Newton iterations, then refine mesh, repeat
            setup_equations(true);

            if (arma::norm(r,"inf") < _tol/2) break; // solution is good enough

            arma::vec du;
            bool solve_success = arma::spsolve(du,J,-r);
            if (not solve_success) {
                _flag = 3;
                break;
            }
            if (du.has_nan() || du.has_inf()) {
                _flag = 2;
                break;
            }

            for (int j=0; j < n; ++j) {
                sol.at(x.at(j)) += du(arma::span(dim*j,dim*(j+1)-1));
            }
        }

        // we count iterations as the number of mesh refinements because the goal of this scheme is to provide
        // a uniform error bound which is achieved by a accurate solution values on a refined mesh. Thus, we allow
        // max_iter mesh refinements, and max_iter + 1 applications of Newton's method to ensure the solution is
        // accurate at the final mesh.
        if (iter >= _max_iter) {
            _flag = 1;
            break;
        }
        
        setup_equations(false);

        auto S = [&](double t, u_long j) -> arma::vec {
            double h = x.at(j) - x.at(j-1);
            const arma::vec& u1 = sol.at(x.at(j));
            const arma::vec& u0 = sol.at(x.at(j-1));
            
            t = (t - x.at(j-1)) / h;
            arma::vec spline = u0 * (1 + 2*t) * std::pow(1 - t,2);
            spline += ff[x.at(j-1)] * h * t * std::pow(1 - t,2);
            spline += u1 * std::pow(t,2) * (3 - 2*t);
            spline += ff[x.at(j)] * h * std::pow(t,2) * (t - 1);
            return spline;
        };

        auto Sp = [&](double t, u_long j) -> arma::vec {
            double h = x.at(j) - x.at(j-1);
            const arma::vec& u1 = sol.at(x.at(j));
            const arma::vec& u0 = sol.at(x.at(j-1));

            t = (t - x.at(j-1)) / h;
            arma::vec spline = u0 * 6*t * (t - 1) / h;
            spline += ff[x.at(j-1)] * (1 - 4*t + 3*std::pow(t,2));
            spline += u1 * 6*t * (1 - t) / h;
            spline += ff[x.at(j)] * t * (3*t - 2);
            return spline;
        };

        auto error = [&](double t, u_long j) -> double {
            return std::pow(arma::norm(Sp(t,j) - f(t, S(t,j))),2);
        };

        arma::vec residuals(n-1);
        for (int j=1; j < n; ++j) {
            double h = x.at(j) - x.at(j-1);
            double x_half = (x.at(j) + x.at(j-1)) / 2;
            double node0 = x_half - std::sqrt(3.0/7)*h/2, node1 = x_half + std::sqrt(3.0/7)*h/2;

            residuals(j-1) = 0.1*std::pow(arma::norm(r(arma::span(dim*(j-1),dim*j-1))),2);
            residuals(j-1) += (49.0/90)*error(node0,j);
            residuals(j-1) += (32.0/35)*error(x_half,j);
            residuals(j-1) += (49.0/90)*error(node1,j);
            residuals(j-1) += 0.1*std::pow(arma::norm(r(arma::span(dim*j,dim*(j+1)-1))),2);
            residuals(j-1) *= h/2;
        }
        residuals = arma::sqrt(residuals);

        arma::uvec idx = arma::find(residuals > _tol);
        if (idx.is_empty()) { // found solution whose error is bounded by tol everywhere
            break;
        } else {
            for (u_int j : idx) {
                x.push_back((x.at(j) + x.at(j+1))/2);
                sol[x.back()] = S(x.back(), j+1); // add new points to grid
            }
            std::stable_sort(x.begin(), x.end());
        }
        iter++;
    }
    _x = arma::conv_to<arma::vec>::from(x);
    _u.set_size(dim, _x.n_elem);
    _du.set_size(dim, _x.n_elem);
    for (u_long i=0; i < x.size(); ++i) {
        _u.col(i) = sol.at(_x(i));
        _du.col(i) = ff.at(_x(i));
    }

    for (u_long i=0; i < dim; ++i) {
        _sol.push_back(hermite_cubic_spline(_x, _u.row(i).as_col(), _du.row(i).as_col(), "boundary"));
    }
    
    _num_iter = iter;
}