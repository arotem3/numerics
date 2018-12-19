#include "numerics.hpp"

//--- nearest neighbor interpolation ---//
//--- x  : x values --------------------//
//----- y  : y values ------------------//
//----- u  : points to evaluate --------//
arma::vec numerics::nearestInterp(const arma::vec& x, const arma::vec& y, const arma::vec& u) {
    int nx = x.n_elem;
    if (arma::size(x) != arma::size(y)) { // dimension error
        std::cerr << "linearInterp() error: interpolation could not be constructed, x and y vectors must be the same length." << std::endl
                  << "\tx has " << x.n_elem << " elements, y has " << y.n_elem << " elements." << std::endl;
        return {NAN};
    }

    for (int i(0); i < nx - 1; ++i) { // repeated x error
        for (int j(i+1); j < nx; ++j) {
            if (std::abs(x(i) - x(j)) < eps()) {
                std::cerr << "linearInterp() error: one or more x values are repeating." << std::endl;
                return {NAN};
            }
        }
    }

    if (!arma::all(u - x(0) >= -0.01) || !arma::all(u - x(nx - 1) <= 0.01)) { // out of bounds error
        std::cerr << "linearInterp() error: atleast one element of u is out of bounds of x." << std::endl;
        return {NAN};
    }

    int nu = u.n_elem;
    arma::vec v(nu, arma::fill::ones);

    arma::uvec a = arma::find( u < (x(0) + x(1))/2 );
    v(a) *= y(1);

    for (int i(1); i < nx-1; ++i) {
        a = (x(i-1)+x(i))/2 <= u && u <= (x(i)+x(i+1))/2;
        a = arma::find( a );
        v(a) = y(i)*arma::ones(arma::size(a));
    }
    a = arma::find( u > (x(nx-1)+x(nx-2))/2 );
    v(a) *= y(nx-1);
    return v;
}

//--- linear interpolation of data points ---//
//----- x  : x values -----------------------//
//----- y  : y values -----------------------//
//----- u  : points to evaluate interpolant -//
arma::vec numerics::linearInterp(const arma::vec& x, const arma::vec& y, const arma::vec& u) {
    int nx = x.n_elem;
    if (arma::size(x) != arma::size(y)) { // dimension error
        std::cerr << "linearInterp() error: interpolation could not be constructed, x and y vectors must be the same length." << std::endl
                  << "\tx has " << x.n_elem << " elements, y has " << y.n_elem << " elements." << std::endl;
        return {NAN};
    }

    for (int i(0); i < nx - 1; ++i) { // repeated x error
        for (int j(i+1); j < nx; ++j) {
            if (std::abs(x(i) - x(j)) < eps()) {
                std::cerr << "linearInterp() error: one or more x values are repeating." << std::endl;
                return {NAN};
            }
        }
    }

    if (!arma::all(u - x(0) >= -0.01) || !arma::all(u - x(nx - 1) <= 0.01)) { // out of bounds error
        std::cerr << "linearInterp() error: atleast one element of u is out of bounds of x." << std::endl;
        return {NAN};
    }
    
    int nu = u.n_elem;
    arma::vec v(nu, arma::fill::zeros);

    for (int i(0); i < nx-1; ++i) {
        arma::uvec a = arma::find( x(i) <= u && u <= x(i+1) );
        v(a) = y(i) * (u(a) - x(i+1))/(x(i) - x(i+1)) + y(i+1) * (u(a) - x(i))/(x(i+1) - x(i));
    }

    return v;
}

//--- lagrange interpolation of data points ---//
//----- x  : x values -------------------------//
//----- y  : y values -------------------------//
//----- u  : points to evaluate interpolant ---//
arma::vec numerics::lagrangeInterp(const arma::vec& x, const arma::vec& y, const arma::vec& u) {
    int nx = x.n_elem;
    if (arma::size(x) != arma::size(y)) { // dimension error
        std::cerr << "lagrangeInterp() error: interpolation could not be constructed, x and y vectors must be the same length." << std::endl
                  << "\tx has " << x.n_elem << " elements, y has " << y.n_elem << " elements." << std::endl;
        return {NAN};
    }

    for (int i(0); i < nx - 1; ++i) { // repeated x error
        for (int j(i+1); j < nx; ++j) {
            if (std::abs(x(i) - x(j)) < eps()) {
                std::cerr << "lagrangeInterp() error: one or more x values are repeating." << std::endl;
                return {NAN};
            }
        }
    }

    if (!arma::all(u - x(0) >= -0.01) || !arma::all(u - x(nx - 1) <= 0.01)) { // out of bounds error
        std::cerr << "lagrangeInterp() error: atleast one element of u is out of bounds of x." << std::endl;
        return {NAN};
    }
    
    int nu = u.n_elem;
    arma::vec v(nu, arma::fill::zeros);

    for (int i(0); i <  nx; ++i) {
        arma::vec P(nu, arma::fill::ones);
        for (int j(0); j < nx; ++j) {
            if (j != i) {
                P %= (u - x(j))/(x(i) - x(j));
            }
        }
        v += y(i) * P;
    }

    return v;
}

//--- multidim lagrange interp ---//
//----- x  : x values ------------//
//----- A  : each col is a y val -//
//----- t  : evaluation pts ------//
arma::mat numerics::LPM(const arma::vec& x, const arma::mat& A, const arma::vec& t) {
    int n = A.n_cols;
    arma::mat S = arma::zeros(t.n_elem,n);

    for (int i(0); i < n; ++i) {
        S.col(i) = lagrangeInterp(x, A.col(i), t);
    }
    return S;
}

arma::mat numerics::LIM(const arma::vec& x, const arma::mat& A, const arma::vec& t) {
    int n = A.n_cols;
    arma::mat S = arma::zeros(t.n_elem,n);

    for (int i(0); i < n; ++i) {
        S.col(i) = linearInterp(x, A.col(i), t);
    }
    return S;
}