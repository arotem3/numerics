#include <numerics.hpp>

/* spectral_deriv(f, a, b, sample_points) : compute spectrally accurate derivative of function over an interval
 * --- f : f(x) function to compute derivative of
 * --- [a,b] : interval over which to evaluate derivative over
 * --- sample_points: number of points to sample (more->more accurate) {default = 50} */
numerics::Polynomial numerics::spectral_deriv(const std::function<double(double)>& f, double a, double b, uint sample_points) {
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

    return Polynomial(0.5*(y+1)*(b-a) + a, W);
}