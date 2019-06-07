#include "ODE.hpp"

/* DIFFMAT4: returns the general 4th order differentiation matrix.
 * --- D : to store mat in.
 * --- x : x values overwhich to calc diff mat
 * --- L,R : limits on x.
 * --- m : number of points. */
void ODE::diffmat4(arma::mat& D, arma::vec& x, double L, double R, uint m) {
    m = m-1;
    x = arma::regspace(0,m)/m; // regspace on [0,1] with m points
    double h = (R - L)/m; // spacing
    x = (R - L) * x + L; // transformation from [0,1] -> [L,R]

    D = arma::zeros(m+1,m+1);
    D.diag(-2) +=  1;
    D.diag(-1) += -8;
    D.diag(1)  +=  8;
    D.diag(2)  += -1;

    D.row(0).head(5) = arma::rowvec({-25, 48, -36, 16, -3});
    D.row(1).head(5) = arma::rowvec({-3, -10, 18, -6, 1});
    D.row(m-1).tail(5) = arma::rowvec({-1, 6, -18, 10, 3});
    D.row(m).tail(5) = arma::rowvec({3, -16, 36, -48, 25});
    D /= 12*h;
}

/* DIFFMAT2: returns the general 2nd order differentiation matrix.
 * --- D : to store mat in.
 * --- x : x values overwhich to calc diff mat
 * --- L,R : limits on x.
 * --- m : number of points. */
void ODE::diffmat2(arma::mat& D, arma::vec& x, double L, double R, uint m) {
    m = m-1;
    x = arma::regspace(0,m)/m; // regspace on [0,1] with m+1 points
    double h = (R - L)/m; // spacing
    x = (R - L) * x + L; // transformation from [0,1] -> [L,R]

    D = arma::zeros(m+1,m+1);
    D.diag(-1) += -1;
    D.diag(1)  +=  1;

    D.row(0).head(3) = arma::rowvec({-3, 4, -1});
    D.row(m).tail(3) = arma::rowvec({1, -4, 3});
    D /= 2*h;
}