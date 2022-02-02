#include <armadillo>
#include "numerics/derivatives.hpp"

using namespace std::complex_literals;

template <typename scalar>
scalar f(scalar x) {
    return std::sin(x);
}

template <typename scalar>
scalar df(scalar x) {
    return std::cos(x);
}

template <typename scalar>
scalar g(const arma::Col<scalar>& x) {
    return arma::prod(x);
}

template <typename scalar>
arma::Col<scalar> dg(const arma::Col<scalar>& x) {
    return arma::prod(x) / x;
}

template <typename scalar>
arma::Mat<scalar> d2g(const arma::Col<scalar>& x) {
    arma::Mat<scalar> B = arma::prod(x) / (x * x.t());
    B.diag().zeros();
    return B;
}

int main() {
    int n_passed = 0;
    int n_failed = 0;

    { // single precision accuracy test
        float e = numerics::deriv<float>(f<float>, 0.5f, 1e-3f) - df<float>(0.5f);
        if (std::abs(e) > 0.01f) {
            std::cout << "deriv failed single precision test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // double precision accuracy test: npt = 1
        double e = numerics::deriv<double>(f<double>, 0.5) - df<double>(0.5);
        if (std::abs(e) > 0.001) {
            std::cout << "deriv failed double precision test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // double precision accuracy test: npt = 2
        double h = std::cbrt(std::numeric_limits<double>::epsilon());
        double e = numerics::deriv<double>(f<double>, 0.5, h, 2) - df<double>(0.5);
        if (std::abs(e) > 0.001) {
            std::cout << "deriv failed double precision, npt = 2, test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // double precision accuracy test: npt = 4
        double h = std::pow(std::numeric_limits<double>::epsilon(), 0.2);
        double e = numerics::deriv<double>(f<double>, 0.5, h, 4) - df<double>(0.5);
        if (std::abs(e) > 0.001) {
            std::cout << "deriv failed double precision, npt = 4, test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // complex double accuracy test
        std::complex<double> e = numerics::deriv<std::complex<double>>(f<std::complex<double>>, 0.5+0.5i) - df<std::complex<double>>(0.5+0.5i);
        if (std::abs(e) > 0.001) {
            std::cout << "deriv failed complex double precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // convergence test
        double e1 = numerics::deriv<double>(f<double>, 0.5, 1e-1) - df<double>(0.5);
        double e2 = numerics::deriv<double>(f<double>, 0.5, 1e-6) - df<double>(0.5);
        if (std::abs(e2) > std::abs(e1)) {
            std::cout << "deriv failed convergence test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // float gradient test
        arma::fvec x = {1.0f, 2.0f, 3.0f};
        float e = arma::norm(numerics::grad(g<float>, x, 1e-3f) - dg(x));
        if (e > 0.01f) {
            std::cout << "grad failed single precision test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // double gradient test
        arma::dvec x = {1.0, 2.0, 3.0};
        double e = arma::norm(numerics::grad(g<double>, x) - dg(x));
        if (e > 0.001) {
            std::cout << "grad failed double precision test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // double jacobian diag test
        arma::dvec x = {1.0, 2.0, 3.0};
        double e = arma::norm(numerics::jacobian_diag(dg<double>, x));
        if (e > 1e-5) {
            std::cout << "jacobian_diag failed double precision test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // double jacobian
        arma::dvec x = {1.0, 2.0, 3.0};
        double e = arma::norm(numerics::jacobian(dg<double>, x) - d2g<double>(x));
        if (e > 1e-3) {
            std::cout << "jacobian failed double precision test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // double directional grad
        arma::fvec x = {1.0f, 2.0f, 3.0f};
        arma::fvec v = {0.5f, 0.8f, -1.0f};

        float e = arma::norm(numerics::directional_grad(dg<float>, x, v) - d2g<float>(x)*v);
        if (e > 0.01) {
            std::cout << "directional_grad failed single precision test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed\n";

    return 0;
}