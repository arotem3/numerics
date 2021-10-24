#include <numerics.hpp>

/* spectral_deriv(f, a, b, sample_points) : compute spectrally accurate derivative of function over an interval
 * --- f : f(x) function to compute derivative of
 * --- [a,b] : interval over which to evaluate derivative over
 * --- sample_points: number of points to sample (more->more accurate) {default = 50} */
numerics::Polynomial numerics::spectral_deriv(const std::function<double(double)>& f, double a, double b, uint sample_points) {
    std::complex<double> i(0,1); // i^2 = -1
    u_long N = sample_points - 1;

    arma::vec y = arma::cos( arma::regspace(0,N)*M_PI/N );
    arma::vec v = y;
    v.for_each([&f,&b,&a](arma::vec::elem_type& u){u = f(0.5*(u+1)*(b-a)+a);});
    
    arma::uvec ii = arma::regspace<arma::uvec>(0,N-1);
    
    arma::vec V = arma::join_cols(v, arma::reverse(v(arma::span(1,N-1))));
    V = arma::real(arma::fft(V));
    
    arma::cx_vec u(2*N);
    u(arma::span(0,N-1)) = i*arma::regspace<arma::cx_vec>(0,N-1);
    u(N) = 0;
    u(arma::span(N+1,2*N-1)) = i*arma::regspace<arma::cx_vec>(1.0-(double)N, -1);
    
    arma::vec W = arma::real(arma::ifft(u%V));
    W.rows(1,N-1) = -W.rows(1,N-1) / arma::sqrt(1 - arma::square(y.rows(1,N-1)));
    W(0) = 0.5*N*V(N) + arma::accu(arma::square(ii) % V.rows(ii)) / N;
    arma::vec j = arma::ones(N); j.rows(arma::regspace<arma::uvec>(1,2,N-1)) *= -1;
    W(N) = 0.5*std::pow(-1,N+1)*N*V(N) + arma::accu(j % arma::square(ii) % V.rows(ii)) / N;
    W = W.rows(0,N);
    W /= (b-a)/2;

    return Polynomial(0.5*(y+1)*(b-a) + a, W);
}