#include "numerics.hpp"

template <typename real>
real f(real x) {
    return std::exp(-x*x);
}

template <typename real>
arma::Col<real> df(arma::Col<real> x) {
    return (-2*x) % arma::exp(-x%x);
}

int main() {
    int n_passed = 0;
    int n_failed = 0;


    { // test 1
        double a = -3, b = 3;
        int m = 32;

        arma::dvec x = {0.5*(a + b)};
        arma::dvec y = df<double>(x);

        numerics::ChebInterp<double> dy = numerics::spectral_deriv(f<double>, a, b, m);
        arma::dvec v = dy(x);
        double e = arma::norm(v - y, "inf");

        if (e > 0.01) {
            std::cout << "spectral_deriv failed double precision test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 2
        float a = -3, b = 3;
        int m = 32;

        arma::fvec x = {0.5f*(a + b)};
        arma::fvec y = df<float>(x);

        numerics::ChebInterp<float> dy = numerics::spectral_deriv(f<float>, a, b, m);
        arma::fvec v = dy(x);
        double e = arma::norm(v - y, "inf");

        if (e > 0.01) {
            std::cout << "spectral_deriv failed single precision test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    {
        double a = -3, b = 3;
        arma::dvec x = {0.5*(a+b)};
        arma::dvec dy = df<double>(x);

        numerics::ChebInterp<double> dy1 = numerics::spectral_deriv(f<double>, a, b, 16);
        numerics::ChebInterp<double> dy2 = numerics::spectral_deriv(f<double>, a, b, 32);

        double e1 = arma::norm(dy - dy1(x));
        double e2 = arma::norm(dy - dy2(x));

        if (e2 > e1) {
            std::cout << "spectral_deriv failed double precision convergence test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed.\n";

    return 0;
}