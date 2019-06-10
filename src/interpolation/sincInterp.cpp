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
            if (std::abs(x(i) - x(j)) < arma::datum::eps) {
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
 * --- [a,b] : interval over which to evaluate derivative over
 * --- sample_points: number of points to sample (more->more accurate) {default = 50} */
numerics::polyInterp numerics::spectral_deriv(const dfunc& f, double a, double b, uint sample_points) {
    arma::cx_double i(0,1); // i^2 = -1
    int N = sample_points - 1;

    arma::vec y = arma::cos( arma::regspace(0,N)*M_PI/N );
    arma::vec v = y;
    v.for_each([&f,&b,&a](arma::vec::elem_type& u){u = f(0.5*(u+1)*(b-a)+a);});
    
    arma::uvec ii = arma::regspace<arma::uvec>(0,N-1);
    v = arma::join_cols(v, arma::reverse(v.rows(1,N-1)));
    v = arma::real(arma::fft(v));
    arma::cx_vec u = i*arma::join_cols(arma::join_cols(arma::regspace(0,N-1), arma::vec({0})), arma::regspace(1-N, -1));
    arma::vec W = arma::real(arma::ifft(u%v));
    W.rows(1,N-1) = -W.rows(1,N-1) / arma::sqrt(1 - arma::square(y.rows(1,N-1)));
    W(0) = 0.5*N*v(N) + arma::accu(arma::square(ii) % v.rows(ii+1)) / N;
    arma::vec j = arma::ones(N); j.rows(arma::regspace<arma::uvec>(1,2,N-1)) *= 2;
    W(N) = 0.5*std::pow(-1,N+1)*N*v(N) + arma::accu(j % arma::square(ii) % v.rows(ii+1)) / N;
    W = W.rows(0,N);
    W /= (b-a)/2;

    polyInterp p(0.5*(y+1)*(b-a) + a, W);
    return p;
}