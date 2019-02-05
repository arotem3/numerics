#include "numerics.hpp"

/* NEARESTINTERP : nearest neighbor interpolation
 * --- x  : x values
 * --- y  : y values
 * --- u  : points to evaluate */
arma::mat numerics::nearestInterp(const arma::vec& x, const arma::mat& y, const arma::vec& u) {
    int nx = x.n_elem;
    if (x.n_elem != y.n_rows) { // dimension error
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
    arma::mat v(nu,y.n_cols, arma::fill::ones);

    arma::uvec a = arma::find( u < (x(0) + x(1))/2 );
    v.each_row(a) %= y.row(1);

    for (int i(1); i < nx-1; ++i) {
        a = (x(i-1)+x(i))/2 <= u && u <= (x(i)+x(i+1))/2;
        a = arma::find( a );
        v.each_row(a) %= y.row(i);
    }
    a = arma::find( u > (x(nx-1)+x(nx-2))/2 );
    v.each_row(a) %= y.row(nx-1);
    return v;
}

/* LINEARINTERP : linear interpolation of data points
 * --- x  : x values
 * --- y  : y values
 * --- u  : points to evaluate interpolant */
arma::mat numerics::linearInterp(const arma::vec& x, const arma::mat& y, const arma::vec& u) {
    int nx = x.n_elem;
    if (x.n_elem != y.n_rows) { // dimension error
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
    arma::mat v(nu, y.n_cols, arma::fill::zeros);

    for (int i(0); i < nx-1; ++i) {
        arma::uvec a = arma::find( x(i) <= u && u <= x(i+1) );
        v.rows(a) = (u(a) - x(i+1))/(x(i) - x(i+1)) * y.row(i) + (u(a) - x(i))/(x(i+1) - x(i)) * y.row(i+1);
    }

    return v;
}

/* LAGRANGEINTERP : lagrange polynomial interpolation of data points
 * --- x  : x values
 * --- y  : y values
 * --- u  : points to evaluate interpolant */
arma::mat numerics::lagrangeInterp(const arma::vec& x, const arma::mat& y, const arma::vec& u) {
    int nx = x.n_elem;
    if (x.n_elem != y.n_rows) { // dimension error
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
    arma::mat v(nu, y.n_cols, arma::fill::zeros);

    for (int i(0); i <  nx; ++i) {
        arma::mat P(nu, y.n_cols, arma::fill::ones);
        for (int j(0); j < nx; ++j) {
            if (j != i) {
                P.each_col() %= (u - x(j))/(x(i) - x(j));
            }
        }
        P.each_row() %= y.row(i);
        v += P;
    }

    return v;
}