#include <numerics.hpp>

/* poisson_helmholtz_2d(x, y, u, f, bc, k, m) : Solves the 2D Poisson/Helmholtz Equation: u_xx + u_yy + k^2 u = f(x,y) with dirichlet boundary conditions.
 * --- f  : f(x,y).
 * --- bc : a function g(x,y) such that if x,y are on the boundary u(x,y) = g(x,y).
 * --- k : square root of Helmholtz eigenvalue.
 * --- m : number of points along each axis. */
void numerics::ode::poisson_helmholtz_2d(arma::mat& X, arma::mat& Y, arma::mat& U,
                              const std::function<arma::mat(const arma::mat&, const arma::mat&)>& f,
                              const std::function<arma::mat(const arma::mat&, const arma::mat&)>& bc,
                              double k,
                              int m
) {
    //--- (1) set up Chebyshev differentiation matrix
    arma::mat D;
    arma::vec x;
    numerics::ode::cheb(D,x,m);
    D *= D; // D^2

    if (X.n_elem < 2) {
        std::cerr << "poisson2d() error: bounds on X not defined.\n";
        return;
    }
    if (Y.n_elem < 2) {
        std::cerr << "poisson2d() error: bounds on Y not defined.\n";
        return;
    }
    double lower_x = X(0), upper_x = X(1), lower_y = Y(0), upper_y = Y(1);

    //--- (2) set up x,y mesh points
    numerics::meshgrid(X, x);
    Y = X.t();
    X = arma::vectorise(X);
    Y = arma::vectorise(Y);

    //--- (3) set up Laplacian matrix operator
    arma::mat I = arma::eye(m, m);
    arma::mat L = arma::kron(I,D) + arma::kron(D,I) + k*k*arma::eye(m*m,m*m);

    //--- (4) apply boundary conditions
    arma::uvec b = arma::find( arma::abs(X) == 1 || arma::abs(Y) == 1 );
    // L.rows(b) = arma::zeros( 4*m, std::pow(m+1,2) );
    L.rows(b).zeros();
    L(b,b) = arma::eye(4*m-4, 4*m-4);

    X = (upper_x - lower_x)*X/2 + (upper_x + lower_x)/2;
    Y = (upper_y - lower_y)*Y/2 + (upper_y + lower_y)/2;
    arma::vec F = f( X, Y );
    F(b) = bc( X(b), Y(b) );
    
    //--- (5) solve the system, reshape, and return
    U = arma::solve(L,F);
    U = arma::reshape(U, m, m);

    X = arma::reshape(X, m, m);
    Y = arma::reshape(Y, m, m);
}