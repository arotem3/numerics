#include <numerics.hpp>

numerics::ode::ODESolution numerics::ode::BVPCheb::ode_solve(const odefunc& f, const boundary_conditions& bc, const arma::vec& x0, const arma::mat& U0) {
    u_long n = x0.n_elem;
    u_long dim = _check_dim(x0, U0);

    _check_x(x0);
    ODESolution sol(dim);
    arma::vec& x = sol._t;
    arma::mat& U = sol._U;

    arma::mat D;
    cheb(D, x, x0.front(), x0.back(), num_pts);
    D = D.rows(1,n-1); // toss first row

    U.set_size(n,dim);
    for (u_long i=0; i < dim; ++i) {
        arma::vec uu;
        arma::interp1(x0, U0.col(i), x, uu, "*linear");
        U.col(i) = uu;
    }
    arma::inplace_trans(U);

    arma::mat DD(n*dim,n*dim), J(n*dim,n*dim);
    DD.rows(0,(n-1)*dim-1) = arma::kron(D,arma::eye(dim,dim));

    arma::mat F, du;
    u_long j = 0;
    do {
        if (j >= _max_iter) {
            std::cerr << "BVPCheb error: too many iterations needed for convergence.\n"
                    << "returning current best estimate.\n"
                    << "least squares error = " << arma::norm(F,"inf") << " > 0\n";
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

        F.set_size(dim,n-1);
        for (int i(0); i < n-1; ++i) {
            F.col(i) = f( x(i+1), U.col(i+1) );
        }

        arma::mat A = D.t()*U - F;
        F = arma::join_cols( A.as_col(), BC );

        J.zeros();
        for (int i(1); i < n; ++i) {
            J.rows((i-1)*dim, (i)*dim-1).cols(i*dim, (i+1)*dim-1) = -approx_jacobian(
                [&](const arma::vec& v) -> arma::vec {
                    return f( x(i), v );
                },
                U.col(i)
            );
        }

        J += DD;

        J.head_cols(dim).rows(n*dim-dim,n*dim-1) = bcJac_L.head_rows(dim);
        J.tail_cols(dim).rows(n*dim-dim,n*dim-1) = bcJac_R.head_rows(dim);

        // solve
        bool solve_success = arma::solve(du, J, -F);
        if (!solve_success || du.has_nan() || du.has_inf()) {
            std::cerr << "BVPCheb error: failed to find update after " << j << " iterations.\n";
            break;
        }
        U += arma::reshape(du,dim,n);
        j++;
    } while (arma::norm(du,"inf") > _tol);
    _num_iter = j;
    arma::inplace_trans(U);
    sol._convert();
    return sol;
}

numerics::ode::ODESolution numerics::ode::BVPCheb::ode_solve(const odefunc& f, const odejacobian& jacobian, const boundary_conditions& bc, const arma::vec& x0, const arma::mat& U0) {
    u_long n = x0.n_elem;
    u_long dim = _check_dim(x0, U0);

    _check_x(x0);
    ODESolution sol(dim);
    arma::vec& x = sol._t;
    arma::mat& U = sol._U;

    arma::mat D;
    cheb(D, x, x0.front(), x0.back(), num_pts);
    D = D.rows(1,n-1); // toss first row

    U.set_size(n,dim);
    for (u_long i=0; i < dim; ++i) {
        arma::vec uu;
        arma::interp1(x0, U0.col(i), x, uu, "*linear");
        U.col(i) = uu;
    }
    arma::inplace_trans(U);

    arma::mat DD(n*dim,n*dim), J(n*dim,n*dim);
    DD.rows(0,(n-1)*dim-1) = arma::kron(D,arma::eye(dim,dim));

    arma::mat F, du;
    u_long j = 0;
    do {
        if (j >= _max_iter) {
            std::cerr << "BVPCheb error: too many iterations needed for convergence.\n"
                    << "returning current best estimate.\n"
                    << "least squares error = " << arma::norm(F,"inf") << " > 0\n";
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

        F.set_size(dim,n-1);
        for (int i(0); i < n-1; ++i) {
            F.col(i) = f( x(i+1), U.col(i+1) );
        }

        arma::mat A = D.t()*U - F;
        F = arma::join_cols( A.as_col(), BC );

        J.zeros();
        for (int i(1); i < n; ++i) {
            J.rows((i-1)*dim, (i)*dim-1).cols(i*dim, (i+1)*dim-1) = -jacobian(x(i),U.col(i));
        }

        J += DD;

        J.head_cols(dim).rows(n*dim-dim,n*dim-1) = bcJac_L.head_rows(dim);
        J.tail_cols(dim).rows(n*dim-dim,n*dim-1) = bcJac_R.head_rows(dim);

        // solve
        bool solve_success = arma::solve(du, J, -F);
        if (!solve_success || du.has_nan() || du.has_inf()) {
            std::cerr << "BVPCheb error: failed to find update after " << j << " iterations.\n";
            break;
        }
        U += arma::reshape(du,dim,n);
        j++;
    } while (arma::norm(du,"inf") > _tol);
    _num_iter = j;
    arma::inplace_trans(U);
    sol._convert();
    return sol;
}