#include <numerics.hpp>

/* ode_solve(x, U, f, bc, guess) : solves general systems of  (potentially) nonlinear boudary value problems.
 * --- x  : x values, must be initialized to [a,b], i.e. the interval where the problem is defined
 * --- U  : u values, will be overwritten.
 * --- f  : u'(x) = f(x,u) is a vector valued function. [u must be a row vector]
 * --- bc : bc(u[0], u[n-1]) == 0, boundary conditions function. */
void numerics::ode::bvp_cheb::ode_solve(arma::vec& x, arma::mat& U,
                    const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                    const std::function<arma::vec(const arma::rowvec&, const arma::rowvec&)>& bc,
                    const std::function<arma::rowvec(double)>& guess
) {
    if (x.n_elem < 2) {
        std::cerr << "bvp_cheb() error: x must be initialized to an interval, but only " << x.n_elem << " points were provided.\n";
        return;
    }
    uint n = num_pts;
    int dim = guess(x(0)).n_elem;

    arma::mat D;
    cheb(D, x, x.min(), x.max(), num_pts);
    D = D.rows(1,n-1); // toss first row

    U.resize(n,dim);
    for (int i=0; i < n; ++i) {
        U.row(i) = guess(x(i));
    }

    arma::mat DD(n*dim,n*dim), J(n*dim,n*dim);
    DD.rows(0,(n-1)*dim-1) = arma::kron(D,arma::eye(dim,dim));

    arma::mat F, du;
    uint j = 0;
    do {
        if (j >= max_iterations) {
            std::cerr << "bvp_cheb() error: too many iterations needed for convergence." << std::endl
                    << "returning current best estimate." << std::endl
                    << "!!!---not necessarily a good estimate---!!!" << std::endl
                    << "least squares error = " << arma::norm(F,"inf") << " > 0" << std::endl;
            break;
        }
        // set up system and jacobian
        arma::vec BC = bc(U.row(0), U.row(n-1));
        if (BC.n_elem < dim) {
            std::cerr << "bvpIIIa() warning: number of boundary conditions (" << BC.n_elem << ") is less than the system dimensions (" << dim << ") suggesting the ODE is underdetermined...returning prematurely\n";
            return;
        } else if (BC.n_elem > dim) {
            std::cerr << "bvpIIIa() warning: number of boundary conditions (" << BC.n_elem << ") exceeds the dimension of the system (" << dim << ") suggesting the ODE is overdetermined... the solver will take only the first " << dim << " conditions.\n";
            return;
        }
        arma::mat bcJac_L = numerics::approx_jacobian([&](const arma::vec& v)->arma::vec{return bc(v.t(),U.row(n-1));}, U.row(0).t());
        arma::mat bcJac_R = numerics::approx_jacobian([&](const arma::vec& v)->arma::vec{return bc(U.row(0),v.t());},U.row(n-1).t());

        F.resize(n-1,dim);
        for (int i(0); i < n-1; ++i) {
            F.row(i) = f( x(i+1), U.row(i+1) );
        }

        arma::mat A = (D*U - F).t();
        F = arma::join_cols( arma::vectorise(A), BC.rows(0,dim-1) );

        J.zeros();
        for (int i(1); i < n; ++i) {
            J.rows((i-1)*dim, (i)*dim-1).cols(i*dim, (i+1)*dim-1) = -approx_jacobian(
                [&](const arma::vec& v) -> arma::vec {
                    return f( x(i), v.t() ).t();
                },
                U.row(i).t()
            );
        }

        J += DD;

        J.head_cols(dim).rows(n*dim-dim,n*dim-1) = bcJac_L.head_rows(dim);
        J.tail_cols(dim).rows(n*dim-dim,n*dim-1) = bcJac_R.head_rows(dim);

        // solve
        bool solve_success = arma::solve(du, J, -F);
        if (!solve_success || du.has_nan() || du.has_inf()) {
            std::cerr << "bvp_cheb() error: failed to find update after " << j << " iterations.\n";
            break;
        }
        U += arma::reshape(du,dim,n).t();
        j++;
    } while (arma::norm(du,"inf") > tol);
    num_iter = j;
}

/* ode_solve(x, U, f, bc, guess) : solves general systems of  (potentially) nonlinear boudary value problems.
 * --- x  : x values, must be initialized (the solver will solve the bvp at these points), and must be sorted.
 * --- U  : u values, must be initialized, zeros should be fine, but an initial guess is ideal.
 * --- f  : u'(x) = f(x,u) is a vector valued function. [u must be a row vector]
 * --- jacobian : J(x,u) jacobian matrix of f(x,u) with respect to u.
 * --- bc : bc(u[0], u[n-1]) == 0, boundary conditions function. */
void numerics::ode::bvp_cheb::ode_solve(arma::vec& x, arma::mat& U,
                    const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                    const std::function<arma::mat(double, const arma::rowvec&)>& jacobian,
                    const std::function<arma::vec(const arma::rowvec&, const arma::rowvec&)>& bc,
                    const std::function<arma::rowvec(double)>& guess
) {
    if (x.n_elem < 2) {
        std::cerr << "bvp_cheb() error: x must be initialized to an interval, but only " << x.n_elem << " points were provided.\n";
        return;
    }
    uint n = num_pts;
    int dim = guess(x(0)).n_elem;

    arma::mat D;
    cheb(D, x, x.min(), x.max(), num_pts);
    D = D.rows(1,n-1); // toss first row

    U.resize(n,dim);
    for (int i=0; i < n; ++i) {
        U.row(i) = guess(x(i));
    }

    arma::mat DD(n*dim,n*dim), J(n*dim,n*dim);
    DD.rows(0,(n-1)*dim-1) = arma::kron(D,arma::eye(dim,dim));

    arma::mat F, du;
    uint j = 0;
    do {
        if (j >= max_iterations) {
            std::cerr << "bvp_cheb() error: too many iterations needed for convergence." << std::endl
                    << "returning current best estimate." << std::endl
                    << "!!!---not necessarily a good estimate---!!!" << std::endl
                    << "least squares error = " << arma::norm(F,"inf") << " > 0" << std::endl;
            break;
        }
        // set up system and jacobian
        arma::vec BC = bc(U.row(0), U.row(n-1));
        if (BC.n_elem < dim) {
            std::cerr << "bvpIIIa() warning: number of boundary conditions (" << BC.n_elem << ") is less than the system dimensions (" << dim << ") suggesting the ODE is underdetermined...returning prematurely\n";
            return;
        } else if (BC.n_elem > dim) {
            std::cerr << "bvpIIIa() warning: number of boundary conditions (" << BC.n_elem << ") exceeds the dimension of the system (" << dim << ") suggesting the ODE is overdetermined... the solver will take only the first " << dim << " conditions.\n";
            return;
        }
        arma::mat bcJac_L = numerics::approx_jacobian([&](const arma::vec& v)->arma::vec{return bc(v.t(),U.row(n-1));}, U.row(0).t());
        arma::mat bcJac_R = numerics::approx_jacobian([&](const arma::vec& v)->arma::vec{return bc(U.row(0),v.t());},U.row(n-1).t());

        F.resize(n-1,dim);
        for (int i(0); i < n-1; ++i) {
            F.row(i) = f( x(i+1), U.row(i+1) );
        }

        arma::mat A = (D*U - F).t();
        F = arma::join_cols( arma::vectorise(A), BC.rows(0,dim-1) );

        J.zeros();
        for (int i(1); i < n; ++i) {
            J.rows((i-1)*dim, (i)*dim-1).cols(i*dim, (i+1)*dim-1) = -jacobian(x(i),U.row(i));
        }

        J += DD;

        J.head_cols(dim).rows(n*dim-dim,n*dim-1) = bcJac_L.head_rows(dim);
        J.tail_cols(dim).rows(n*dim-dim,n*dim-1) = bcJac_R.head_rows(dim);

        // solve
        bool solve_success = arma::solve(du, J, -F);
        if (!solve_success || du.has_nan() || du.has_inf()) {
            std::cerr << "bvp_cheb() error: failed to find update after " << j << " iterations.\n";
            break;
        }
        U += arma::reshape(du,dim,n).t();
        j++;
    } while (arma::norm(du,"inf") > tol);
    num_iter = j;
}