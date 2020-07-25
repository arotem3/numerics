#include <numerics.hpp>

numerics::ode::ODESolution numerics::ode::BVPCheb::ode_solve(const odefunc& f, const boundary_conditions& bc, const arma::vec& x0, const arma::mat& U0) {
    u_long dim = _check_dim(x0, U0);

    _check_x(x0);
    ODESolution sol(dim);
    arma::vec& x = sol._t;
    arma::mat& U = sol._U;

    arma::mat D;
    cheb(D, x, x0.front(), x0.back(), num_pts);
    arma::mat DD = arma::kron(D,arma::eye(dim,dim));

    U.set_size(dim, num_pts);
    for (u_long i=0; i < dim; ++i) {
        arma::vec uu;
        if (U0.n_rows == num_pts) arma::interp1(x0, U0.col(i), x, uu, "*linear");
        else arma::interp1(x0, U0.row(i).as_col(), x, uu, "*linear");
        U.row(i) = uu.as_row();
    }

    u_long row1, row2, col1, col2; // for accessing submatrix views

    arma::mat F, du;
    arma::mat J((num_pts+1)*dim, num_pts*dim);
    u_long j = 0;
    do {
        if (j >= _max_iter) {
            sol._flag = 1;
            break;
        }
        // set up system and jacobian
        arma::vec BC = bc(U.col(0), U.col(num_pts-1));
        if (BC.n_elem != dim) {
            throw std::runtime_error("number of boundary conditions (=" + std::to_string(BC.n_elem) + ") does not match the system dimensions (=" + std::to_string(dim) + ")");
        }
        arma::mat bcJac_L = numerics::approx_jacobian(
            [&](const arma::vec& v) -> arma::vec {
                return bc(v,U.col(num_pts-1));
            },
            U.col(0)
        );
        arma::mat bcJac_R = numerics::approx_jacobian(
            [&](const arma::vec& v) -> arma::vec {
                return bc(U.col(0),v);
            },
            U.col(num_pts-1)
        );

        F.set_size(dim,num_pts);
        for (int i=0; i < num_pts; ++i) {
            F.col(i) = f( x(i), U.col(i) );
        }

        arma::mat A = U*D.t() - F;
        F = arma::join_cols( A.as_col(), BC );

        J.zeros();
        for (int i=0; i < num_pts; ++i) {
            J.submat(i*dim, i*dim, (i+1)*dim-1, (i+1)*dim-1) =  -approx_jacobian(
                [&](const arma::vec& v) -> arma::vec {
                    return f(x(i), v);
                },
                U.col(i)
            );
        }
        J.rows(0, num_pts*dim-1) += DD;

        col1 = 0; col2 = dim-1;
        row1 = num_pts*dim; row2 = (num_pts+1)*dim-1;
        J.submat(row1, col1, row2, col2) = bcJac_L;
        col1 = J.n_cols - dim; col2 = J.n_cols - 1;
        J.submat(row1, col1, row2, col2) = bcJac_R;

        // solve
        bool solve_success = arma::solve(du, J, -F);
        if (not solve_success) {
            sol._flag = 3;
            break;
        }
        if (du.has_nan() || du.has_inf()) {
            sol._flag = 2;
            break;
        }
        U += arma::reshape(du,dim,num_pts);
        j++;
    } while (arma::norm(du,"inf") > _tol);
    _num_iter = j;
    arma::inplace_trans(U);
    sol._prepare();
    return sol;
}

numerics::ode::ODESolution numerics::ode::BVPCheb::ode_solve(const odefunc& f, const odejacobian& jacobian, const boundary_conditions& bc, const arma::vec& x0, const arma::mat& U0) {
    u_long dim = _check_dim(x0, U0);

    _check_x(x0);
    ODESolution sol(dim);
    arma::vec& x = sol._t;
    arma::mat& U = sol._U;

    arma::mat D;
    cheb(D, x, x0.front(), x0.back(), num_pts);
    arma::mat DD = arma::kron(D,arma::eye(dim,dim));

    U.set_size(dim, num_pts);
    for (u_long i=0; i < dim; ++i) {
        arma::vec uu;
        if (U0.n_rows == num_pts) arma::interp1(x0, U0.col(i), x, uu, "*linear");
        else arma::interp1(x0, U0.row(i).as_col(), x, uu, "*linear");
        U.row(i) = uu.as_row();
    }

    u_long row1, row2, col1, col2; // for accessing submatrix views

    arma::mat F, du;
    arma::mat J((num_pts+1)*dim, num_pts*dim);
    u_long j = 0;
    do {
        if (j >= _max_iter) {
            sol._flag = 1;
            break;
        }
        // set up system and jacobian
        arma::vec BC = bc(U.col(0), U.col(num_pts-1));
        if (BC.n_elem != dim) {
            throw std::runtime_error("number of boundary conditions (=" + std::to_string(BC.n_elem) + ") does not match the system dimensions (=" + std::to_string(dim) + ")");
        }
        arma::mat bcJac_L = numerics::approx_jacobian(
            [&](const arma::vec& v) -> arma::vec {
                return bc(v,U.col(num_pts-1));
            },
            U.col(0)
        );
        arma::mat bcJac_R = numerics::approx_jacobian(
            [&](const arma::vec& v) -> arma::vec {
                return bc(U.col(0),v);
            },
            U.col(num_pts-1)
        );

        F.set_size(dim,num_pts);
        for (int i=0; i < num_pts; ++i) {
            F.col(i) = f( x(i), U.col(i) );
        }

        arma::mat A = U*D.t() - F;
        F = arma::join_cols( A.as_col(), BC );

        J.zeros();
        for (int i=0; i < num_pts; ++i) {
            J.submat(i*dim, i*dim, (i+1)*dim-1, (i+1)*dim-1) =  -jacobian(x(i),U.col(i));
        }
        J.rows(0, num_pts*dim-1) += DD;

        col1 = 0; col2 = dim-1;
        row1 = num_pts*dim; row2 = (num_pts+1)*dim-1;
        J.submat(row1, col1, row2, col2) = bcJac_L;
        col1 = J.n_cols - dim; col2 = J.n_cols - 1;
        J.submat(row1, col1, row2, col2) = bcJac_R;

        // solve
        bool solve_success = arma::solve(du, J, -F);
        if (not solve_success) {
            sol._flag = 3;
            break;
        }
        if (du.has_nan() || du.has_inf()) {
            sol._flag = 2;
            break;
        }
        U += arma::reshape(du,dim,num_pts);
        j++;
    } while (arma::norm(du,"inf") > _tol);
    _num_iter = j;
    arma::inplace_trans(U);
    sol._prepare();
    return sol;
}