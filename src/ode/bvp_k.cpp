#include <numerics.hpp>

numerics::ode::ODESolution numerics::ode::BVPk::ode_solve(const odefunc& f, const boundary_conditions& bc, const arma::vec& x0, const arma::mat& U0) {
    u_long n = x0.n_elem;
    u_long dim = _check_dim(x0, U0);

    _check_x(x0);

    ODESolution sol(dim);
    arma::vec& x = sol._t;
    arma::mat& U = sol._U;
    x = x0;
    if (U.n_rows == n) U = U0.t();
    else U = U0;
    
    arma::sp_mat D;
    diffmat(D, x, 1, k);

    arma::sp_mat DD = arma::kron(D, arma::speye(dim,dim));
    
    arma::mat F, du;
    u_long j = 0;
    u_long row1, row2, col1, col2; // for accessing submatrix views
    do {
        if (j >= _max_iter) {
            sol._flag = 1;
            break;
        }
        // set up system and jacobian
        arma::vec BC = bc(U.col(0), U.col(n-1));
        if (BC.n_elem != dim) {
            throw std::runtime_error("number of boundary conditions (=" + std::to_string(BC.n_elem) + ") does not match the system dimensions (=" + std::to_string(dim) + ")");
        }

        arma::mat bcJac_L = numerics::approx_jacobian(
            [&](const arma::vec& v) -> arma::vec {
                return bc(v,U.col(n-1));
            },
            U.col(0)
        );
        arma::mat bcJac_R = numerics::approx_jacobian(
            [&](const arma::vec& v) -> arma::vec {
                return bc(U.col(0),v);
            },
            U.col(n-1)
        );
        
        F.set_size(dim,n);
        for (u_long i=0; i < n; ++i) {
            F.col(i) = f( x(i), U.col(i) );
        }
        arma::mat A = U*D.t() - F;
        F = arma::join_cols( A.as_col(), BC );

        arma::sp_mat J((n+1)*dim,n*dim);
        for (u_long i=0; i < n; ++i) {
            J.submat(i*dim, i*dim, (i+1)*dim-1, (i+1)*dim-1) =  -approx_jacobian(
                [&](const arma::vec& v) -> arma::vec {
                    return f(x(i), v);
                },
                U.col(i)
            );
        }
        J.rows(0,n*dim-1) += DD;
        
        col1 = 0; col2 = dim-1;
        row1 = n*dim; row2 = (n+1)*dim-1;
        J.submat(row1, col1, row2, col2) = bcJac_L;
        col1 = J.n_cols - dim; col2 = J.n_cols - 1;
        J.submat(row1, col1, row2, col2) = bcJac_R;

        F = J.t()*F;
        J = J.t()*J;

        // solve
        arma::superlu_opts opts;
        opts.symmetric = true;
        bool solve_success = arma::spsolve(du, J, -F, "superlu", opts);
        if (not solve_success) {
            sol._flag = 3;
            break;
        }
        if (du.has_nan() || du.has_inf()) {
            sol._flag = 2;
            break;
        }
        U += arma::reshape(du,dim,n);
        j++;
    } while (arma::norm(du,"inf") > _tol);
    _num_iter = j;
    arma::inplace_trans(U);
    sol._prepare();
    return sol;
}

numerics::ode::ODESolution numerics::ode::BVPk::ode_solve(const odefunc& f, const odejacobian& jacobian, const boundary_conditions& bc, const arma::vec& x0, const arma::mat& U0) {
    u_long n = x0.n_elem;
    u_long dim = _check_dim(x0, U0);

    _check_x(x0);

    ODESolution sol(dim);
    arma::vec& x = sol._t;
    arma::mat& U = sol._U;
    x = x0;
    if (U.n_rows == n) U = U0.t();
    else U = U0;
    
    arma::sp_mat D;
    diffmat(D, x, 1, k);

    arma::sp_mat DD = arma::kron(D, arma::speye(dim,dim));
    
    arma::mat F, du;
    u_long j = 0;
    u_long row1, row2, col1, col2; // for accessing submatrix views
    do {
        if (j >= _max_iter) {
            sol._flag = 1;
            break;
        }
        // set up system and jacobian
        arma::vec BC = bc(U.col(0), U.col(n-1));
        if (BC.n_elem != dim) {
            throw std::runtime_error("number of boundary conditions (=" + std::to_string(BC.n_elem) + ") does not match the system dimensions (=" + std::to_string(dim) + ")");
        }

        arma::mat bcJac_L = numerics::approx_jacobian(
            [&](const arma::vec& v) -> arma::vec {
                return bc(v,U.col(n-1));
            },
            U.col(0)
        );
        arma::mat bcJac_R = numerics::approx_jacobian(
            [&](const arma::vec& v) -> arma::vec {
                return bc(U.col(0),v);
            },
            U.col(n-1)
        );
        
        F.set_size(dim,n);
        for (u_long i=0; i < n; ++i) {
            F.col(i) = f( x(i), U.col(i) );
        }
        arma::mat A = U*D.t() - F;
        F = arma::join_cols( A.as_col(), BC );

        arma::sp_mat J((n+1)*dim,n*dim);
        for (u_long i=0; i < n; ++i) {
            J.submat(i*dim, i*dim, (i+1)*dim-1, (i+1)*dim-1) = -jacobian(x(i), U.col(i));
        }
        J.rows(0,n*dim-1) += DD;
        
        col1 = 0; col2 = dim-1;
        row1 = n*dim; row2 = (n+1)*dim-1;
        J.submat(row1, col1, row2, col2) = bcJac_L;
        col1 = J.n_cols - dim; col2 = J.n_cols - 1;
        J.submat(row1, col1, row2, col2) = bcJac_R;

        F = J.t()*F;
        J = J.t()*J;

        // solve
        arma::superlu_opts opts;
        opts.symmetric = true;
        bool solve_success = arma::spsolve(du, J, -F, "superlu", opts);
        if (not solve_success) {
            sol._flag = 3;
            break;
        }
        if (du.has_nan() || du.has_inf()) {
            sol._flag = 2;
            break;
        }
        U += arma::reshape(du,dim,n);
        j++;
    } while (arma::norm(du,"inf") > _tol);
    _num_iter = j;
    arma::inplace_trans(U);
    sol._prepare();
    return sol;
}