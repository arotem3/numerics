#include <numerics.hpp>

numerics::ode::ODESolution numerics::ode::BVP3a::ode_solve(const odefunc& f, const boundary_conditions& bc, const arma::vec& x0, const arma::mat& U0) {
    u_long n = x0.n_elem;
    u_long dim = _check_dim(x0, U0);

    _check_x(x0);
    ODESolution sol(dim);
    arma::vec& x = sol._t;
    arma::mat& U = sol._U;
    x = x0;
    if (U.n_rows == n) U = U0.t();
    else U = U0;

    u_long k = 0;
    arma::sp_mat J(dim*n,dim*n);
    arma::mat F(n,dim), dU;
    do {
        if (k >= _max_iter) {
            sol._flag = 1;
            break;
        }
        arma::vec BC = bc(U.col(0),U.col(n-1));
        arma::mat bcJac_L = numerics::approx_jacobian(
            [&](const arma::vec& v) -> arma::vec {
                return bc(v, U.col(n-1));
            },
            U.col(0)
        );
        arma::mat bcJac_R = numerics::approx_jacobian(
            [&](const arma::vec& v) -> arma::vec {
                return bc(U.col(0), v);
            },
            U.col(n-1)
        );

        arma::vec fi,fip1,y;
        arma::mat dfdy, dfdu, dydu;
        F.set_size(dim,n);

        dfdu = numerics::approx_jacobian(
            [&](const arma::vec& v) -> arma::vec {
                return f(x(0),v);
            },
            U.col(0)
        );
        fi = f(x(0), U.col(0));

        for (u_long i=0; i < n-1; ++i) {
            double h = x(i+1) - x(i);
            double s = x(i) + h/2;
            fip1 = f(x(i+1),U.col(i+1));
            y = (U.col(i+1) + U.col(i))/2 + h/8 * (fi - fip1);
            
            F.col(i+1) = U.col(i+1) - U.col(i) - h/6 * (fi + 4*f(s,y) + fip1);
            
            dfdy = numerics::approx_jacobian(
                [&](const arma::vec& v) -> arma::vec {
                    return f(s,v);
                }, y, h*h
            );
            dydu = 0.5*arma::eye(dim,dim) + h/8 * dfdu;

            J.submat((i+1)*dim, i*dim, (i+2)*dim-1, (i+1)*dim-1) = -arma::eye(dim,dim) - h/6 * (dfdu + 4*dfdy*dydu);
            
            dfdu = numerics::approx_jacobian(
                [&](const arma::vec& v) -> arma::vec {
                    return f(x(i+1),v);
                }, U.col(i+1), h*h
            );
            dydu = 0.5*arma::eye(dim,dim) + h/8 * dfdu;

            J.submat((i+1)*dim,(i+1)*dim,(i+2)*dim-1,(i+2)*dim-1) = arma::eye(dim,dim) - h/6 * (dfdu + 4*dfdy*dydu);
            fi = fip1;
        }
        F = F.as_col();
        int n_bc = BC.n_elem;
        if (n_bc != dim) {
            throw std::runtime_error("number of boundary conditions (=" + std::to_string(BC.n_elem) + ") does not match the system dimensions (=" + std::to_string(dim) + ")");
        }
        F.rows(0,dim-1) = BC;
        J.submat(0, 0, dim-1, dim-1) = bcJac_L;
        J.submat(0, J.n_cols-dim, dim-1, J.n_cols-1) = bcJac_R;

        bool solve_success = arma::spsolve(dU,J,-F);
        if (not solve_success) {
            sol._flag = 3;
            break;
        }
        if (dU.has_nan() || dU.has_inf()) {
            sol._flag = 2;
            break;
        }
        U += arma::reshape(dU,dim,n);
        k++;
    } while (arma::norm(dU,"inf") > _tol);
    _num_iter = k;
    arma::inplace_trans(U);
    sol._prepare();
    return sol;
}

numerics::ode::ODESolution numerics::ode::BVP3a::ode_solve(const odefunc& f, const odejacobian& jacobian, const boundary_conditions& bc, const arma::vec& x0, const arma::mat& U0) {
    u_long n = x0.n_elem;
    u_long dim = _check_dim(x0, U0);

    _check_x(x0);
    ODESolution sol(dim);
    arma::vec& x = sol._t;
    arma::mat& U = sol._U;
    x = x0;
    if (U.n_rows == n) U = U0.t();
    else U = U0;

    u_long k = 0;
    arma::sp_mat J(dim*n,dim*n);
    arma::mat F(n,dim), dU;
    do {
        if (k >= _max_iter) {
            sol._flag = 1;
            break;
        }
        arma::vec BC = bc(U.col(0),U.col(n-1));
        arma::mat bcJac_L = numerics::approx_jacobian(
            [&](const arma::vec& v) -> arma::vec {
                return bc(v, U.col(n-1));
            },
            U.col(0)
        );
        arma::mat bcJac_R = numerics::approx_jacobian(
            [&](const arma::vec& v) -> arma::vec {
                return bc(U.col(0), v);
            },
            U.col(n-1)
        );

        arma::vec fi,fip1,y;
        arma::mat dfdy, dfdu, dydu;
        F.set_size(dim,n);

        dfdu = jacobian(x(0),U.col(0));
        fi = f(x(0), U.col(0));

        for (u_long i=0; i < n-1; ++i) {
            double h = x(i+1) - x(i);
            double s = x(i) + h/2;
            fip1 = f(x(i+1),U.col(i+1));
            y = (U.col(i+1) + U.col(i))/2 + h/8 * (fi - fip1);
            
            F.col(i+1) = U.col(i+1) - U.col(i) - h/6 * (fi + 4*f(s,y) + fip1);
            
            dfdy = jacobian(s,y);
            dydu = 0.5*arma::eye(dim,dim) + h/8 * dfdu;

            J.submat((i+1)*dim, i*dim, (i+2)*dim-1, (i+1)*dim-1) = -arma::eye(dim,dim) - h/6 * (dfdu + 4*dfdy*dydu);
            
            dfdu = jacobian(x(i+1),U.col(i+1));
            dydu = 0.5*arma::eye(dim,dim) + h/8 * dfdu;

            J.submat((i+1)*dim,(i+1)*dim,(i+2)*dim-1,(i+2)*dim-1) = arma::eye(dim,dim) - h/6 * (dfdu + 4*dfdy*dydu);
            fi = fip1;
        }
        F = F.as_col();
        int n_bc = BC.n_elem;
        if (n_bc != dim) {
            throw std::runtime_error("number of boundary conditions (=" + std::to_string(BC.n_elem) + ") does not match the system dimensions (=" + std::to_string(dim) + ")");
        }
        F.rows(0,dim-1) = BC;
        J.submat(0, 0, dim-1, dim-1) = bcJac_L;
        J.submat(0, J.n_cols-dim, dim-1, J.n_cols-1) = bcJac_R;

        bool solve_success = arma::spsolve(dU,J,-F);
        if (not solve_success) {
            sol._flag = 3;
            break;
        }
        if (dU.has_nan() || dU.has_inf()) {
            sol._flag = 2;
            break;
        }
        U += arma::reshape(dU,dim,n);
        k++;
    } while (arma::norm(dU,"inf") > _tol);
    _num_iter = k;
    arma::inplace_trans(U);
    sol._prepare();
    return sol;
}