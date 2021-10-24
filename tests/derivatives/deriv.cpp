// #include "numerics.hpp"

#include <armadillo>

template <typename real>
real f(real x) {
    return std::sin(x);
}

template <typename real>
real df(real x) {
    return std::cos(x);
}

template <typename real>
real g(const arma::Col<real>& x) {
    return arma::prod(x);
}

template <typename real>
arma::Col<real> dg(const arma::Col<real>& x) {
    return arma::prod(x) / x;
}

template <typename real>
arma::Mat<real> d2g(const arma::Col<real>& x) {
    arma::Mat<real> B = arma::prod(x) / (x * x.t());
    B.diag().zeros();
    return B;
}

int main() {
    int n_passed = 0;
    int n_failed = 0;

    { // test 1
        float e = numerics::deriv<float>(f<float>, 0.5f, 1e-3f) - df<float>(0.5f);
        if (std::abs(e) > 0.01f) {
            std::cout << "deriv failed single precision test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 2
        double e = numerics::deriv<double>(f<double>, 0.5) - df<double>(0.5);
        if (std::abs(e) > 0.001) {
            std::cout << "deriv failed double precision test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 3
        double e1 = numerics::deriv<double>(f<double>, 0.5, 1e-1) - df<double>(0.5);
        double e2 = numerics::deriv<double>(f<double>, 0.5, 1e-6) - df<double>(0.5);
        if (std::abs(e2) > std::abs(e1)) {
            std::cout << "deriv failed convergence test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 4
        arma::fvec x = {1.0f, 2.0f, 3.0f};
        float e = arma::norm(numerics::grad(g<float>, x, 1e-3f) - dg(x));
        if (e > 0.01f) {
            std::cout << "grad failed single precision test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 5
        arma::dvec x = {1.0, 2.0, 3.0};
        double e = arma::norm(numerics::grad(g<double>, x) - dg(x));
        if (e > 0.001) {
            std::cout << "grad failed double precision test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 6
        arma::dvec x = {1.0, 2.0, 3.0};
        double e = arma::norm(numerics::jacobian_diag(dg<double>, x));
        if (e > 1e-5) {
            std::cout << "jacobian_diag failed double precision test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 7
        arma::dvec x = {1.0, 2.0, 3.0};
        double e = arma::norm(numerics::jacobian(dg<double>, x) - d2g<double>(x));
        if (e > 1e-3) {
            std::cout << "jacobian failed double precision test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 8
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