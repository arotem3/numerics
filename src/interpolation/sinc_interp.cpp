#include "numerics.hpp"

/* sinc_interp(x, y, u) : interpolate data points with sinc functions
 * --- x : uniformly spaced domain data
 * --- y : points to interpolate
 * --- u : points to evaluate interpolation on */
arma::mat numerics::sinc_interp(const arma::vec& x, const arma::mat& y, const arma::vec& u) {
    int n = x.n_elem;
    if (x.n_elem != y.n_rows) { // dimension error
        std::cerr << "sinc_interp() error: interpolation could not be constructed, x and y vectors must be the same length." << std::endl
                  << "\tx has " << x.n_elem << " elements, y has " << y.n_elem << " elements." << std::endl;
        return {NAN};
    }

    for (int i(0); i < n - 1; ++i) { // repeated x error
        for (int j(i+1); j < n; ++j) {
            if (std::abs(x(i) - x(j)) < arma::datum::eps) {
                std::cerr << "sinc_interp() error: one or more x values are repeating." << std::endl;
                return {NAN};
            }
        }
    }

    double h = x(1) - x(0);
    arma::mat v(u.n_elem, y.n_cols, arma::fill::zeros);
    for (int i(0); i < n; ++i) {
        arma::mat s = arma::repmat(y.row(i), u.n_elem, 1);
        s.each_col() %= arma::sinc(  (u - x(i))/h  );
        v += s;
    }

    return v;
}