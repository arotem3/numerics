#include "ODE.hpp"

//--- solves general systems of nonlinear boudary value problems ---//
//----- f  : u'(x) = f(x,u) is a vector valued function ------------//
//----- bc : boundary conditions struct ----------------------------//
//-------- bc.xL : lower limit of x --------------------------------//
//-------- bc.xR : upper limit of x --------------------------------//
//-------- bc.func : func(uL,uR) = 0 the boundary conditions -------//
//----- guess : guess(x), guess of solution as a function of x -----//
//----- opts  : struct containing options and results for bvp ------//
ODE::dsolnp ODE::bvp(odefun f, bcfun bc, soln_init guess, bvp_opts& opts) {
    int m = opts.num_points;
    int n = guess({bc.xL}).n_elem;
    arma::mat D;
    arma::vec x;
    cheb(D, x, bc.xL, bc.xR, m-1);
    
    auto ff = [&](const arma::vec& u) -> arma::vec {
        arma::mat U = arma::reshape(u,m,n);
        arma::rowvec BC = bc.func(u.row(0), u.row(m-1));
        arma::mat A = D;
        
        arma::mat F = arma::zeros(m,n);
        for (int i(0); i < m; ++i) {
            F.row(i) = f( x(i), U.row(i) );
        }
        A = A*U - F;
        arma::vec z = arma::join_cols( arma::vectorise(A), arma::vectorise(BC) );
        return z;
    }; // wrapper for root finding function
    arma::vec U0 = arma::vectorise( guess(x) );
    if (opts.solver == numerics::LMLSQR) numerics::lmlsqr(ff, U0, opts.lsqropts);
    else numerics::broyd(ff, U0, opts.nlnopts);

    arma::mat U = arma::reshape(U0,m,n);

    dsolnp Soln;
    Soln.independent_var_values = x;
    Soln.solution_values = U;
    Soln.soln = numerics::polyInterp(x,U);
    return Soln;
}

ODE::dsolnp ODE::bvp(odefun f, bcfun bc, soln_init guess) {
    bvp_opts opts;
    return bvp(f,bc,guess,opts);
}