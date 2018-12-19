#include "numerics.hpp"

//--- cubic interpolation of a single variable ---//
//----- x  : independent variable ----------------//
//----- y  : dependent variable ------------------//
//----- t  : points to evaluate interpolation on -//
//----- s  : return var, array of evaluated pts --//
arma::vec cubicInterp(const arma::vec& x, const arma::vec& y, const arma::vec& t) {
    int n = arma::size(x)(0) - 1;
    arma::mat h = arma::zeros(n,1); //  step sizes
    arma::mat A = arma::zeros(n+1,n+1);
    arma::mat RHS = arma::zeros(n+1,1);
    arma::mat b = arma::zeros(n,1);
    arma::mat d = arma::zeros(n,1);
    int t_length = arma::size(t)(0);
    arma::mat s = arma::zeros(t_length,1);

    for (int i(1); i < n+1; ++i) {
        h(i-1) = x(i) - x(i-1); 
    }

    A(0,0) = 1;
    A(n,n) = 1;

    for (int i(1); i < n; ++i) {
        A(i,i-1) = h(i-1);
        A(i,i) = 2 * (h(i) + h(i-1));
        A(i,i+1) = h(i);

        RHS(i) = 3 * (y(i+1) - y(i))/h(i) - 3 * (y(i) - y(i-1))/h(i-1);
    }

    arma::vec c = solve(A,RHS);

    for (int i(0); i < n; ++i) {
        b(i) = (y(i+1) - y(i))/h(i) - h(i)*(2*c(i) + c(i+1))/3;
        d(i) = (c(i+1) - c(i))/(3*h(i));
    }

    for (int i(0); i < t_length; ++i) {
        for (int j(0); j < n; ++j) {
            if (t(i) >= x(j) && t(i) <= x(j+1)) {
                s(i) = y(j) + b(j)*(t(i) - x(j)) + c(j)*pow(t(i) - x(j),2) + d(j)*pow(t(i)-x(j),3);
            }
        }
    }

    return s;
}

//--- cubic interpolation of multiple variables against a single parameter ---//
//----- X  : each colomn is a data array for a each variable -----------------//
//----- n  : the number of points to evaluate the interpolation on -----------//
//----- paramFit : first col is the param vals, the rest are the evaluations -//
arma::mat paramIterp(const arma::mat& X, int n) {
    int x_size = arma::size(X)(0);
    int dim = arma::size(X)(1);

    arma::vec t = arma::linspace(0, 1, x_size);
    arma::vec s = arma::linspace(0, 1, n);

    arma::mat paramFit = arma::zeros(x_size, dim+1);
    paramFit.col(0) = s;

    for (int i(1); i < dim+1; ++i) {
        paramFit.col(i) = cubicInterp(t, X.col(i), s);
    }

    return paramFit;
} // THIS FUNCTION IS BAD PRACTICE!

//--- bicubic spline interpolation of a surface z = f(x,y) ---//
//----- x  : domain x values ---------------------------------//
//----- y  : domain y values ---------------------------------//
//----- z  : range z values ----------------------------------//
//----- xRange : a single/set of x values to eval ------------//
//----- yRange : a single/set of y values to eval ------------//
arma::mat bicubic(arma::mat& x, arma::mat& y, arma::mat& z, arma::vec& xRange, arma::vec& yRange) {
    int m = arma::size(z)(0);
    int n = arma::size(xRange)(0);
    int p = arma::size(yRange)(0);
    arma::mat temp(m,n,arma::fill::zeros);
    for (int i(0); i < m; ++i) {
        temp.row(i) = cubicInterp(x, z.row(i).t(), xRange).t();
    }
    arma::mat w(p,n,arma::fill::zeros);
    for (int i(0); i < p; ++i) {
        w.col(i) = cubicInterp(y, temp.col(i), yRange);
    }
    return w;
}

arma::mat bicubic(arma::mat& z, double scale) {
    int m = arma::size(z)(0);
    int n = arma::size(z)(1);
    int mNew = scale*m;
    int nNew = scale*n;

    arma::mat temp(m, mNew, arma::fill::zeros);
    for (int i(0); i < m; ++i) {
        temp.row(i) = cubicInterp(arma::linspace(0,1,m), z.row(i).t(), arma::linspace(0,1,mNew));
    }
    
    arma::mat w(nNew, mNew, arma::fill::zeros);
    for (int i(0); i < nNew; ++i) {
        w.col(i) = cubicInterp(arma::linspace(0,1,n), temp.col(i), arma::linspace(0,1,nNew));
    }
    return w;
}

double bicubic(arma::mat& x, arma::mat& y, arma::mat& z, double u, double v) {
    arma::vec xRange = {u};
    arma::vec yRange = {v};
    arma::mat w = bicubic(x, y, z, xRange, yRange);
    return w(0);
}