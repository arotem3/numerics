#include <numerics.hpp>

/* ode_solve(x, U, f, bc, guess) : solves general systems of  (potentially) nonlinear boudary value problems.
 * --- x  : x values will be stored here.
 * --- U  : u values will be stored here.
 * --- f  : u'(x) = f(x,u) is a vector valued function. [u must be a row vector]
 * --- bc : boundary conditions class
 * ----- bc.xL : lower limit of x
 * ----- bc.xR : upper limit of x
 * ----- bc.condition(uL,uR) : boundary condition such the condition(u(xL), u(xR)) == 0.
 * --- guess : guess(x) ~= u(x) initial guess of solution as a function of x. */
void numerics::ode::bvp::ode_solve(arma::vec& x, arma::mat& U,
                              const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                              const boundary_conditions& bc,
                              const std::function<arma::mat(const arma::vec&)>& guess) {
    int m = num_points;
    int n = guess({bc.xL}).n_elem; // system dimension
    arma::mat D;
    if (order == bvp_solvers::CHEBYSHEV) {
        cheb(D, x, bc.xL, bc.xR, m);
    } else if (order == bvp_solvers::SECOND_ORDER) diffmat2(D, x, bc.xL, bc.xR, m);
    else diffmat4(D, x, bc.xL, bc.xR, m);
    
    auto ff = [&](const arma::vec& u) -> arma::vec {
        arma::mat U = arma::reshape(u,m,n);
        arma::rowvec BC = bc.condition(U.row(0), U.row(m-1));
        
        arma::mat F = arma::zeros(m,n);
        for (int i(0); i < m; ++i) {
            F.row(i) = f( x(i), U.row(i) );
        }

        arma::mat A = D*U - F;
        arma::vec z = arma::join_cols( arma::vectorise(A), BC.t() );
        return z;
    };

    auto J = [&](const arma::vec& u) -> arma::mat {
        arma::mat U = arma::reshape(u,m,n);
        arma::mat DF = arma::zeros(m*n,m*n);
        for (int i(0); i < m; ++i) {
            auto fff = [&](const arma::vec& v) -> arma::vec {
                return f( x(i), v.t() ).t();
            };
            arma::mat JJ = approx_jacobian( fff, U.row(i).t() );
            for (int j(0); j < n; ++j) {
                for (int k(0); k < n; ++k) {
                    DF(m*j+i, m*k+i) = JJ(j,k);
                }
            }
        }
        DF += arma::kron(arma::eye(n,n), D);

        auto bc_wrapper = [&](const arma::vec& v) -> arma::vec {
            arma::mat V = arma::reshape(v,2,n).t(); // fills by row...
            arma::mat bc_result = bc.condition(V.row(0),V.row(1));
            return arma::vectorise(bc_result);
        };
        arma::vec vv = arma::join_rows( U.row(0), U.row(m-1) ).t();
        arma::mat X = approx_jacobian(bc_wrapper, vv);
        int p = X.n_rows;
        arma::mat bc_jac = arma::zeros(p,m*n);
        short j = 0;
        for (int i(0); i < n; ++i) {
            bc_jac.col(i*m) = X.col(j);
            j++;
            bc_jac.col((i+1)*m-1) = X.col(j);
            j++;
        }
        DF = arma::join_cols(DF,bc_jac);
        return DF;
    };
    
    U = arma::vectorise( guess(x) );
    arma::mat Jinv = arma::pinv( J(U) );
    arma::vec F,F1 = ff(U),du,JiY;
    uint k = 0;
    do {
        if (k >= max_iterations) {
            std::cerr << "bvp() failed: too many iterations needed for convergence." << std::endl
                    << "returning current best estimate." << std::endl
                    << "!!!---not necessarily a good estimate---!!!" << std::endl
                    << "least squares error = " << arma::norm(ff(x),"inf") << " > 0" << std::endl;
            break;
        }
        F = F1;
        du = -(Jinv * F);
        U += du;
        F1 = ff(U);

        JiY = Jinv*(F1 - F);
        Jinv += (du - JiY)*du.t()*Jinv/arma::dot(du, JiY);
        if ( Jinv.has_nan() ) {
            Jinv = arma::pinv( J(U) );
        }

        k++;
    } while (arma::norm(du,"inf") > tol);
    num_iter = k;
    U = arma::reshape(U,m,n);
}

/* ode_solve(x, U, f, bc, guess) : solves general systems of  (potentially) nonlinear boudary value problems.
 * --- x  : x values will be stored here.
 * --- U  : u values will be stored here.
 * --- f  : u'(x) = f(x,u) is a vector valued function. [u must be a row vector]
 * --- jacobian : J(x,u) jacobian matrix of f(x,u) with respect to u.
 * --- bc : boundary conditions class
 * ----- bc.xL : lower limit of x
 * ----- bc.xR : upper limit of x
 * ----- bc.condition(uL,uR) : boundary condition such the condition(u(xL), u(xR)) == 0.
 * --- guess : guess(x) ~= u(x) initial guess of solution as a function of x. */
void numerics::ode::bvp::ode_solve(arma::vec& x, arma::mat& U,
                              const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                              const std::function<arma::mat(double, const arma::rowvec&)>& jacobian,
                              const boundary_conditions& bc,
                              const std::function<arma::mat(const arma::vec&)>& guess) {
    int m = num_points;
    int n = guess({bc.xL}).n_elem; // system dimension
    arma::mat D;
    if (order == bvp_solvers::CHEBYSHEV) {
        cheb(D, x, bc.xL, bc.xR, m);
    } else if (order == bvp_solvers::SECOND_ORDER) diffmat2(D, x, bc.xL, bc.xR, m);
    else diffmat4(D, x, bc.xL, bc.xR, m);
    
    auto ff = [&](const arma::vec& u) -> arma::vec {
        arma::mat U = arma::reshape(u,m,n);
        arma::rowvec BC = bc.condition(U.row(0), U.row(m-1));
        
        arma::mat F = arma::zeros(m,n);
        for (int i(0); i < m; ++i) {
            F.row(i) = f( x(i), U.row(i) );
        }

        arma::mat A = D*U - F;
        arma::vec z = arma::join_cols( arma::vectorise(A), BC.t() );
        return z;
    };

    auto J = [&](const arma::vec& u) -> arma::mat {
        arma::mat U = arma::reshape(u,m,n);
        arma::mat DF = arma::zeros(m*n,m*n);
        for (int i(0); i < m; ++i) {
            arma::mat JJ = jacobian( x(i), U.row(i) );
            for (int j(0); j < n; ++j) {
                for (int k(0); k < n; ++k) {
                    DF(m*j+i, m*k+i) = JJ(j,k);
                }
            }
        }
        DF += arma::kron(arma::eye(n,n), D);

        auto bc_wrapper = [&](const arma::vec& v) -> arma::vec {
            arma::mat V = arma::reshape(v,2,n).t(); // fills by row...
            arma::mat bc_result = bc.condition( V.row(0), V.row(1) );
            return arma::vectorise(bc_result);
        };
        arma::vec vv = arma::join_rows( U.row(0), U.row(m-1) ).t();
        arma::mat X = approx_jacobian(bc_wrapper, vv);
        int p = X.n_rows;
        arma::mat bc_jac = arma::zeros(p,m*n);
        short j = 0;
        for (int i(0); i < n; ++i) {
            bc_jac.col(i*m) = X.col(j);
            j++;
            bc_jac.col((i+1)*m-1) = X.col(j);
            j++;
        }
        DF = arma::join_cols(DF,bc_jac);
        return DF;
    };
    
    U = arma::vectorise( guess(x) );
    arma::mat Jinv = arma::pinv( J(U) );
    arma::vec F,F1 = ff(U),du,JiY;
    uint k = 0;
    do {
        if (k >= max_iterations) {
            std::cerr << "bvp() failed: too many iterations needed for convergence." << std::endl
                    << "returning current best estimate." << std::endl
                    << "!!!---not necessarily a good estimate---!!!" << std::endl
                    << "least squares error = " << arma::norm(ff(x),"inf") << " > 0" << std::endl;
            break;
        }
        F = F1;
        du = -(Jinv * F);
        U += du;
        F1 = ff(U);

        JiY = Jinv*(F1 - F);
        Jinv += (du - JiY)*du.t()*Jinv/arma::dot(du, JiY);
        if ( Jinv.has_nan() ) {
            Jinv = arma::pinv( J(U) );
        }

        k++;
    } while (arma::norm(du,"inf") > tol);
    num_iter = k;
    U = arma::reshape(U,m,n);
}