#include "ODE.hpp"

/* POISSON2D : Solves the 2D Poisson's Equation: L*u(x,y) = f(x,y) with simple BCs.
 * --- f  : f(x,y).
 * --- bc : boundary conditions struct. */
ODE::soln_2d ODE::poisson2d(const pde2fun& f, const bcfun_2d& bc, uint num_pts) {
    auto change_x = [&](const arma::vec& x) -> arma::vec { // [-1,1] -> [lower_x, upper_x]
        return (bc.upper_x - bc.lower_x)*x/2 + (bc.upper_x + bc.lower_x)/2;
    };
    auto change_y = [&](const arma::vec& y) -> arma::vec { // [-1,1] -> [lower_y, upper_y]
        return (bc.upper_y - bc.lower_y)*y/2 + (bc.upper_y + bc.lower_y)/2;
    };
    int m = num_pts - 1;
    //--- (1) set up Chebyshev differentiation matrix
    arma::mat D;
    arma::vec x;
    cheb(D,x,num_pts);
    D = D*D; // D^2

    //--- (2) set up x,y mesh points
    arma::mat xx; numerics::meshgrid(xx, x);
    arma::mat yy = xx.t();
    xx = arma::vectorise(xx);
    yy = arma::vectorise(yy);

    //--- (3) set up Laplacian matrix operator
    arma::mat I = arma::eye(m+1, m+1);
    arma::mat L = arma::kron(I,D) + arma::kron(D,I);

    //--- (4) apply boundary conditions
    arma::uvec b = arma::find( arma::abs(xx) == 1 || arma::abs(yy) == 1 );
    L.rows(b) = arma::zeros( 4*m, std::pow(m+1,2) );
    L(b,b) = arma::eye(4*m, 4*m);

    arma::vec F = f( change_x(xx), change_y(yy) );
    F(b) =    (xx(b) == -1) % bc.lower_x_bc( change_y(yy(b)) )
            + (xx(b) ==  1) % bc.upper_x_bc( change_y(yy(b)) )
            + (yy(b) == -1) % bc.lower_y_bc( change_x(xx(b)) )
            + (yy(b) ==  1) % bc.upper_y_bc( change_x(xx(b)) );
    
    //--- (5) solve the system, reshape, and return
    arma::mat u = arma::solve(L,F);
    u = arma::reshape(u, m+1, m+1);

    xx = arma::reshape(xx, m+1, m+1);
    yy = xx.t();

    xx = (bc.upper_x - bc.lower_x)*xx + bc.lower_x + bc.upper_x;
    xx /= 2.0;
    yy = (bc.upper_y - bc.lower_y)*yy + bc.lower_y + bc.upper_y;
    yy /= 2.0;

    soln_2d soln;
    soln.X = xx;
    soln.Y = yy;
    soln.U = u;
    return soln;
}