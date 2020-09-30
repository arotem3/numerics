#include "numerics.hpp"

/* lagrange_interp(x, y, u, normalize) : lagrange polynomial interpolation of data points
 * --- x  : x values
 * --- y  : y values
 * --- u  : points to evaluate interpolant
 * --- normalize : whether to improve conditioning of interpolant by scaling distant according to exp(-x^2) */
arma::mat numerics::lagrange_interp(const arma::vec& x, const arma::mat& y, const arma::vec& u) {
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
    
    int nu = u.n_elem;
    arma::mat v(nu, y.n_cols, arma::fill::zeros);
    double var;

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