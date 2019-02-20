#include "ODE.hpp"

/* BVP : solves general systems of  (potentially) nonlinear boudary value problems.
 * --- f  : u'(x) = f(x,u) is a vector valued function.
 * --- bc : boundary conditions struct.
 * ----- bc.xL : lower limit of x
 * ----- bc.xR : upper limit of x
 * ----- bc.func : func(uL,uR) = 0 the boundary conditions.
 * --- guess : guess(x), guess of solution as a function of x.
 * --- opts  : struct containing options and results for bvp. */
ODE::dsolnp ODE::bvp(const odefun& f, const bcfun& bc, const soln_init& guess, bvp_opts& opts) {
    int m = opts.num_points;
    int n = guess({bc.xL}).n_elem;
    arma::mat D;
    arma::vec x;
    if (opts.order == bvp_solvers::CHEBYSHEV) cheb(D, x, bc.xL, bc.xR, m-1);
    else if (opts.order == bvp_solvers::SECOND_ORDER) diffmat2(D, x, bc.xL, bc.xR, m-1);
    else diffmat4(D, x, bc.xL, bc.xR, m-1);
    
    numerics::vector_func ff = [&](const arma::vec& u) -> arma::vec {
        arma::mat U = arma::reshape(u,m,n);
        arma::rowvec BC = bc.func(U.row(0), U.row(m-1));
        
        arma::mat F = arma::zeros(m,n);
        for (int i(0); i < m; ++i) {
            F.row(i) = f( x(i), U.row(i) );
        }

        arma::mat A = D*U - F;
        arma::vec z = arma::join_cols( arma::vectorise(A), BC.t() );
        return z;
    }; // wrapper for root finding function

    numerics::vec_mat_func J = [&](const arma::vec& u) -> arma::mat {
        arma::mat U = arma::reshape(u,m,n);
        arma::mat DF = arma::zeros(m*n,m*n);
        if (opts.jacobian_func != nullptr) { // Jacobian provided
            for (int i(0); i < m; ++i) {
                arma::mat JJ = opts.jacobian_func->operator()( x(i), U.row(i) );
                for (int j(0); j < n; ++j) {
                    for (int k(0); k < n; ++k) {
                        DF(m*j+i, m*k+i) = JJ(j,k);
                    }
                }
            }
        } else { // no jacobian provided
            for (int i(0); i < m; ++i) {
                auto fff = [&](const arma::vec& v) -> arma::vec {
                    return f( x(i), v.t() ).t();
                };
                arma::mat JJ;
                numerics::approx_jacobian(fff,JJ,U.row(i).t(),1e-4);
                for (int j(0); j < n; ++j) {
                    for (int k(0); k < n; ++k) {
                        DF(m*j+i, m*k+i) = JJ(j,k);
                    }
                }
            }
        }
        DF = DF + arma::kron(arma::eye(n,n), D);

        auto bc_wrapper = [&](const arma::vec& v) -> arma::vec {
            arma::mat V = arma::reshape(v,2,n).t(); // fills by row...
            return bc.func(V.row(0), V.row(1)).t();
        };
        arma::vec vv = arma::join_rows( U.row(0), U.row(m-1) ).t();
        arma::mat X;
        numerics::approx_jacobian(bc_wrapper, X, vv);
        arma::mat bc_jac = arma::zeros(n,m*n);
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
    
    arma::vec U0 = arma::vectorise( guess(x) );
    opts.nlnopts.jacobian_func = &J;
    numerics::broyd(ff, U0, opts.nlnopts);

    arma::mat U = arma::reshape(U0,m,n);

    dsolnp Soln;
    Soln.independent_var_values = x;
    Soln.solution_values = U;
    if (opts.order == bvp_solvers::CHEBYSHEV) Soln.soln = numerics::polyInterp(x,U);
    return Soln;
}

/* BVP : solves general systems of  (potentially) nonlinear boudary value problems.
 * currently uses spectral solver only, so ideal for smooth functions only.
 * --- f  : u'(x) = f(x,u) is a vector valued function.
 * --- bc : boundary conditions struct.
 * ----- bc.xL : lower limit of x
 * ----- bc.xR : upper limit of x
 * ----- bc.func : func(uL,uR) = 0 the boundary conditions.
 * --- guess : guess(x), guess of solution as a function of x. */
ODE::dsolnp ODE::bvp(const odefun& f, const bcfun& bc, const soln_init& guess) {
    bvp_opts opts;
    return bvp(f,bc,guess,opts);
}