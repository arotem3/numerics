#include <iostream>

#include <armadillo>
#include "numerics/optimization/lmlsqr.hpp"

template <class vec, typename scalar=typename vec::value_type>
vec f(const vec& v)
{
    scalar x = v[0], y = v[1];
    vec fv(2);
    fv[0] = x*x + y*y - 1.0;
    fv[1] = x + y - std::sqrt(2);
    return fv;
}

template <typename scalar>
arma::Mat<scalar> J(const arma::Col<scalar>& v)
{
    scalar x = v[0], y = v[1];
    arma::Mat<scalar> J = {
        {scalar(2.0)*x, scalar(2.0)*y},
        {scalar(1.0), scalar(1.0)}
    };

    return J;
}

using numerics::optimization::lmlsqr;
using numerics::optimization::OptimizationOptions;
using namespace std::complex_literals;

int main()
{
    int n_passed = 0;
    int n_failed = 0;

    { // double precision arma no jacobian
        arma::vec x = {-0.01, 0.0}; // jac is singular at 0,0 so start away from 0,0

        OptimizationOptions<double> opts;
        double tol = opts.ftol * std::max(1.0, arma::norm(f(x)));
        lmlsqr(x, f<arma::vec>, opts);
        if (arma::norm(f(x)) > tol) {
            std::cout << "lmlsqr failed armadillo double precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // single precision arma no jacobian
        arma::fvec x = {-0.01f, 0.0f};
        
        OptimizationOptions<float> opts;
        float tol = opts.ftol * std::max(1.0f, arma::norm(f(x)));
        lmlsqr(x, f<arma::fvec>, opts);
        if (arma::norm(f(x)) > tol) {
            std::cout << "lmlsqr failed armadillo single precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // complex double precision w/ jacobian
        arma::cx_vec x = {-0.01+0.01i, 0.0+0.0i};

        OptimizationOptions<double> opts;
        double tol = opts.ftol * std::max(1.0, arma::norm(f(x)));
        lmlsqr(x, f<arma::cx_vec>, opts);
        if (arma::norm(f(x)) > tol) {
            std::cout << "lmlsqr failed complex double precision\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // double precision arma w/ jacobian
        arma::vec x = {-0.01, 0.0};
        
        OptimizationOptions<double> opts;
        lmlsqr(x, f<arma::vec>, J<double>, opts);
        double tol = opts.ftol * std::max(1.0, arma::norm(f(x)));
        if (arma::norm(f(x)) > tol) {
            std::cout << "lmlsqr failed armadillo double precision w/ jacobian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // single precision arma w/ jacobian
        arma::fvec x = {-0.01f, 0.0f};
        
        OptimizationOptions<float> opts;
        float tol = opts.ftol * std::max(1.0f, arma::norm(f(x)));
        lmlsqr(x, f<arma::fvec>, J<float>, opts);
        if (arma::norm(f(x)) > tol) {
            std::cout << "lmlsqr failed armadillo single precision w/ jacobian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // complex double precision w/ jacobian
        arma::cx_vec x = {-0.01+0.01i, 0.0+0.0i};

        OptimizationOptions<double> opts;
        double tol = opts.ftol * std::max(1.0, arma::norm(f(x)));
        lmlsqr(x, f<arma::cx_vec>, J<arma::cx_double>, opts);
        if (arma::norm(f(x)) > tol) {
            std::cout << "lmlsqr failed double precision w/ jacobian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed\n";

    return 0;
}