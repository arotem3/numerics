#include <numerics.hpp>

void numerics::ode::BVPCheb::solve_bvp(const odefunc& f, const boundary_conditions& bc, const arma::vec& x0, const arma::mat& U0) {
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

void numerics::ode::BVPCheb::solve_bvp(const odefunc& f, const odejacobian& jacobian, const boundary_conditions& bc, const arma::vec& x0, const arma::mat& U0) {
    u_long dim = _check_dim(x0, U0);

    _check_x(x0);

    arma::mat D;
    cheb(D, _x, x0.front(), x0.back(), num_pts);
    arma::mat DD = arma::kron(D,arma::eye(dim,dim));

    _u.set_size(dim, num_pts);
    for (u_long i=0; i < dim; ++i) {
        arma::vec uu;
        if (U0.n_rows == num_pts) arma::interp1(x0, U0.col(i), _x, uu, "*linear");
        else arma::interp1(x0, U0.row(i).as_col(), _x, uu, "*linear");
        _u.row(i) = uu.as_row();
    }

    u_long row1, row2, col1, col2; // for accessing submatrix views

    arma::mat F(dim,num_pts), du;
    arma::mat J((num_pts+1)*dim, num_pts*dim);
    u_long j = 0;
    do {
        if (j >= _max_iter) {
            _flag = 1;
            break;
        }
        
        // set up system and jacobian
        arma::vec BC = bc(_u.col(0), _u.col(num_pts-1));
        if (BC.n_elem != dim) {
            throw std::runtime_error("number of boundary conditions (=" + std::to_string(BC.n_elem) + ") does not match the system dimensions (=" + std::to_string(dim) + ")");
        }
        arma::mat bcJac_L = numerics::approx_jacobian(
            [&](const arma::vec& v) -> arma::vec {
                return bc(v,_u.col(num_pts-1));
            },
            _u.col(0)
        );
        arma::mat bcJac_R = numerics::approx_jacobian(
            [&](const arma::vec& v) -> arma::vec {
                return bc(_u.col(0),v);
            },
            _u.col(num_pts-1)
        );

        for (int i=0; i < num_pts; ++i) {
            F.col(i) = f( _x(i), _u.col(i) );
        }

        arma::mat A = _u*D.t() - F;

        arma::vec RHS = arma::join_cols( A.as_col(), BC );

        J.zeros();
        for (int i=0; i < num_pts; ++i) {
            J.submat(i*dim, i*dim, (i+1)*dim-1, (i+1)*dim-1) =  -jacobian(_x(i),_u.col(i));
        }
        J.rows(0, num_pts*dim-1) += DD;

        col1 = 0; col2 = dim-1;
        row1 = num_pts*dim; row2 = (num_pts+1)*dim-1;
        J.submat(row1, col1, row2, col2) = bcJac_L;
        col1 = J.n_cols - dim; col2 = J.n_cols - 1;
        J.submat(row1, col1, row2, col2) = bcJac_R;

        // solve
        bool solve_success = arma::solve(du, J, -RHS);
        if (not solve_success) {
            _flag = 3;
            break;
        }
        if (du.has_nan() || du.has_inf()) {
            _flag = 2;
            break;
        }
        _u += arma::reshape(du,dim,num_pts);
        j++;
    } while (arma::abs(du).max() > _tol);
    _num_iter = j;

    for (u_int i=0; i < num_pts; ++i) {
        F.col(i) = f(x(i), _u.col(i));
    }
    _du = std::move(F);

    for (u_long i=0; i < dim; ++i) {
        _sol.push_back(Polynomial(_x, _u.row(i).as_col()));
    }
}