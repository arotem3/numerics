#include "numerics.hpp"

/* SINCINTERP : interpolate data points with sinc functions
 * --- x : uniformly spaced domain data
 * --- y : points to interpolate
 * --- u : points to evaluate interpolation on */
arma::mat numerics::sincInterp(const arma::vec& x, const arma::mat& y, const arma::vec& u) {
    int n = x.n_elem;
    if (x.n_elem != y.n_rows) { // dimension error
        std::cerr << "sincInterp() error: interpolation could not be constructed, x and y vectors must be the same length." << std::endl
                  << "\tx has " << x.n_elem << " elements, y has " << y.n_elem << " elements." << std::endl;
        return {NAN};
    }

    for (int i(0); i < n - 1; ++i) { // repeated x error
        for (int j(i+1); j < n; ++j) {
            if (std::abs(x(i) - x(j)) < eps()) {
                std::cerr << "sincInterp() error: one or more x values are repeating." << std::endl;
                return {NAN};
            }
        }
    }

    if (arma::any(u - x(0) < -0.01) || !arma::all(u - x(n - 1) <= 0.01)) { // out of bounds error
        std::cerr << "sincInterp() error: atleast one element of u is out of bounds of x." << std::endl;
        return {NAN};
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

/* SPECTRAL_DERIV : compute spectrally accurate derivative of function over an interval
 * --- f : f(x) function to compute derivative of
 * --- x : points to evaluate derivative at
 * --- sample_points: number of points to sample (more->more accurate) */
arma::vec numerics::specral_deriv(const dfunc& f, arma::vec& x, int sample_points) {
    double a = x(0);
    int m = x.n_elem;
    double b = x(m-1);
    double h = (b - a)/sample_points;
    x = h * arma::regspace(1,sample_points) + a;
    std::cout << h << std::endl;
    arma::vec up = x;
    up.for_each(  [f](arma::vec::elem_type& u){u = f(u);}  );
    arma::cx_colvec up_hat = arma::fft(up);
    arma::vec k = arma::join_cols(
            arma::join_cols(arma::regspace(0,sample_points/2-1),arma::vec({0})),
            arma::regspace(-sample_points/2+1,-1)
    );
    up_hat %= arma::cx_double(0,1) * k / h;
    up = arma::real(arma::ifft(up_hat)) * (2*M_PI / sample_points);
    return up;
}