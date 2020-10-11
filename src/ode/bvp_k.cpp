#include <numerics.hpp>

void numerics::ode::BVPk::solve_bvp(const odefunc& f, const boundary_conditions& bc, const arma::vec& x0, const arma::mat& U0) {
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

void numerics::ode::BVPk::solve_bvp(const odefunc& f, const odejacobian& jacobian, const boundary_conditions& bc, const arma::vec& x0, const arma::mat& U0) {
    u_long n = x0.n_elem;
    u_long dim = _check_dim(x0, U0);

    _check_x(x0);

    _x = x0;
    if (U0.n_rows == n) _u = U0.t();
    else _u = U0;
    
    arma::sp_mat D;
    diffmat(D, _x, 1, k);

    arma::sp_mat DD = arma::kron(D, arma::speye(dim,dim));
    
    arma::mat F(dim,n), du;
    u_long j = 0;
    u_long row1, row2, col1, col2; // for accessing submatrix views
    do {
        if (j >= _max_iter) {
            _flag = 1;
            break;
        }

        // set up system and jacobian
        arma::vec BC = bc(_u.col(0), _u.col(n-1));
        if (BC.n_elem != dim) {
            throw std::runtime_error("number of boundary conditions (=" + std::to_string(BC.n_elem) + ") does not match the system dimensions (=" + std::to_string(dim) + ")");
        }

        arma::mat bcJac_L = numerics::approx_jacobian(
            [&](const arma::vec& v) -> arma::vec {
                return bc(v,_u.col(n-1));
            },
            _u.col(0)
        );
        arma::mat bcJac_R = numerics::approx_jacobian(
            [&](const arma::vec& v) -> arma::vec {
                return bc(_u.col(0),v);
            },
            _u.col(n-1)
        );

        for (u_long i=0; i < n; ++i) {
            F.col(i) = f( _x(i), _u.col(i) );
        }

        arma::mat A = _u*D.t() - F;

        arma::vec RHS = arma::join_cols( A.as_col(), BC );

        arma::sp_mat J((n+1)*dim,n*dim);
        for (u_long i=0; i < n; ++i) {
            J.submat(i*dim, i*dim, (i+1)*dim-1, (i+1)*dim-1) = -jacobian(_x(i), _u.col(i));
        }
        J.rows(0,n*dim-1) += DD;
        
        col1 = 0; col2 = dim-1;
        row1 = n*dim; row2 = (n+1)*dim-1;
        J.submat(row1, col1, row2, col2) = bcJac_L;
        col1 = J.n_cols - dim; col2 = J.n_cols - 1;
        J.submat(row1, col1, row2, col2) = bcJac_R;

        RHS = J.t()*RHS;
        J = J.t()*J;

        // solve
        arma::superlu_opts opts;
        opts.symmetric = true;
        bool solve_success = arma::spsolve(du, J, -RHS, "superlu", opts);
        if (not solve_success) {
            _flag = 3;
            break;
        }
        if (du.has_nan() || du.has_inf()) {
            _flag = 2;
            break;
        }
        _u += arma::reshape(du,dim,n);
        j++;
    } while (arma::abs(du).max() > _tol);
    
    _num_iter = j;
    
    for (u_int i=0; i < n; ++i) {
        F.col(i) = f(_x(i), _u.col(i));
    }
    _du = std::move(F);

    for (u_int i=0; i < dim; ++i) {
        _sol.push_back(hermite_cubic_spline(_x,_u.row(i).as_col(),_du.row(i).as_col(),"boundary"));
    }
}