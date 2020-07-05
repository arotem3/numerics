#include "numerics.hpp"

/* lagrange_interp(x, y, u, normalize) : lagrange polynomial interpolation of data points
 * --- x  : x values
 * --- y  : y values
 * --- u  : points to evaluate interpolant
 * --- normalize : whether to improve conditioning of interpolant by scaling distant according to exp(-x^2) */
arma::mat numerics::lagrange_interp(const arma::vec& x, const arma::mat& y, const arma::vec& u, bool normalize) {
    int nx = x.n_elem;
    if (x.n_elem != y.n_rows) { // dimension error
        std::cerr << "lagrange_interp() error: interpolation could not be constructed, x and y vectors must be the same length." << std::endl
                  << "\tx has " << x.n_elem << " elements, y has " << y.n_elem << " elements." << std::endl;
        return {NAN};
    }

    for (int i(0); i < nx - 1; ++i) { // repeated x error
        for (int j(i+1); j < nx; ++j) {
            if (std::abs(x(i) - x(j)) < arma::datum::eps) {
                std::cerr << "lagrange_interp() error: one or more x values are repeating." << std::endl;
                return {NAN};
            }
        }
    }

    if ((u.min() < x.min()-0.01) || (u.max() > x.max()+0.01)) { // out of bounds error
        std::cerr << "lagrange_interp() error: atleast one element of u is out of bounds of x." << std::endl;
        return {NAN};
    }
    
    int nu = u.n_elem;
    arma::mat v(nu, y.n_cols, arma::fill::zeros);
    double var;
    if (normalize) var = 0.5*arma::range(x)/x.n_elem;

    for (int i(0); i <  nx; ++i) {
        arma::mat P(nu, y.n_cols, arma::fill::ones);
        for (int j(0); j < nx; ++j) {
            if (j != i) {
                P.each_col() %= (u - x(j))/(x(i) - x(j));
            }
        }
        if (normalize) {
            P.each_col() %= arma::exp(-arma::square(u - x(i))/var);
        }
        P.each_row() %= y.row(i);
        v += P;
    }

    return v;
}