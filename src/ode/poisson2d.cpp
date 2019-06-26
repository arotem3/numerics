#include <numerics.hpp>

/* poisson2d(x, y, u, f, bc, m) : Solves the 2D Poisson's Equation: u_xx + u_yy = f(x,y) with dirichlet boundary conditions.
 * --- f  : f(x,y).
 * --- bc : boundary_conditions_2d class to specify bounds on x and y, as well as provide a function g(x,y) such that if x,y are on the boundary u(x,y) = g(x,y). */
void numerics::ode::poisson2d(arma::mat& X, arma::mat& Y, arma::mat& U,
                              const std::function<arma::mat(const arma::mat&, const arma::mat&)>& f,
                              const boundary_conditions_2d& bc, int m) {
    //--- (1) set up Chebyshev differentiation matrix
    arma::mat D;
    arma::vec x;
    numerics::ode::cheb(D,x,m);
    D *= D; // D^2

    //--- (2) set up x,y mesh points
    numerics::meshgrid(X, x);
    Y = X.t();
    X = arma::vectorise(X);
    Y = arma::vectorise(Y);

    //--- (3) set up Laplacian matrix operator
    arma::mat I = arma::eye(m, m);
    arma::mat L = arma::kron(I,D) + arma::kron(D,I);

    //--- (4) apply boundary conditions
    arma::uvec b = arma::find( arma::abs(X) == 1 || arma::abs(Y) == 1 );
    // L.rows(b) = arma::zeros( 4*m, std::pow(m+1,2) );
    L.rows(b).zeros();
    L(b,b) = arma::eye(4*m-4, 4*m-4);

    X = (bc.upper_x - bc.lower_x)*X/2 + (bc.upper_x + bc.lower_x)/2;
    Y = (bc.upper_y - bc.lower_y)*Y/2 + (bc.upper_y + bc.lower_y)/2;
    arma::vec F = f( X, Y );
    F(b) = bc.dirichlet_condition( X(b), Y(b) );
    
    //--- (5) solve the system, reshape, and return
    U = arma::solve(L,F);
    U = arma::reshape(U, m, m);

    X = arma::reshape(X, m, m);
    Y = arma::reshape(Y, m, m);
}