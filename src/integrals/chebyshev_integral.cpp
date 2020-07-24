#include <numerics.hpp>

double numerics::chebyshev_integral(const std::function<double(double)>& f, double a, double b, uint m) {
    arma::vec x = arma::regspace(0,m-1)/(m-1);
    x = 0.5*(1+arma::cos(M_PI * x))*(b-a) + a;

    arma::vec y = x;
    y.for_each([&f](arma::vec::elem_type& u) -> void {u = f(u);});
    y = arma::join_cols(y, arma::flipud(y.rows(1,y.n_rows-2)));
    arma::vec c = arma::real(arma::fft(y));
    c = arma::flipud(c.rows(0,m-1))/(m-1);
    c(0) /= 2;
    c(m-1) /= 2;

    arma::vec p1 = arma::zeros(m); p1(m-1) = 1;
    arma::vec p2 = arma::zeros(m); p2(m-2) = 1;
    arma::vec p = c(m-1)*p1 + c(m-2)*p2;
    for (uint i=m-3; i > 0; --i) {
        arma::vec temp = p2;
        p2 = arma::shift(2*p2,-1) - p1;
        p1 = temp;
        p += c(i) * p2;
    }
    p = polyint(p);
    return (b-a)/2 * arma::as_scalar(arma::diff(arma::polyval(p,arma::vec({-1,1}))));
}