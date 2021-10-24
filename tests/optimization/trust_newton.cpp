#include <iostream>

#include <armadillo>
#include "numerics/optimization/trust_newton.hpp"

template <class vec, typename real=typename vec::value_type>
vec f(const vec& v)
{
    real x = v[0], y = v[1];
    vec fv(2);
    fv[0] = x*x + y*y - 1;
    fv[1] = x + y - std::sqrt(2);
    return fv;
}

template <typename real>
arma::Mat<real> J(const arma::Col<real>& v)
{
    real x = v[0], y = v[1];
    arma::Mat<real> J = {
        {2*x, 2*y},
        {1, 1}
    };

    return J;
}

using numerics::optimization::trust_newton;
using numerics::optimization::OptimizationOptions;

int main()
{
    int n_passed = 0;
    int n_failed = 0;

    { // test 1: double precision arma no jacobian
        arma::vec x = {-0.01, 0.0}; // jac is singular at 0,0 so start away from 0,0
        
        OptimizationOptions<double> opts;
        trust_newton(x, f<arma::vec>, opts);
        if (arma::norm(f(x)) > opts.ftol) {
            std::cout << "trust_newton failed armadillo double precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 2: single precision arma no jacobian
        arma::fvec x = {-0.01f, 0.0f};
        
        OptimizationOptions<float> opts;
        trust_newton(x, f<arma::fvec>, opts);
        if (arma::norm(f(x)) > opts.ftol) {
            std::cout << "trust_newton failed armadillo single precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 3: double precision arma w/ jacobian
        arma::vec x = {-0.01, 0.0};
        
        OptimizationOptions<double> opts;
        trust_newton(x, f<arma::vec>, J<double>, opts);
        if (arma::norm(f(x)) > opts.ftol) {
            std::cout << "trust_newton failed armadillo double precision w/ jacobian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 4: single precision arma w/ jacobian
        arma::fvec x = {-0.01f, 0.0f};
        
        OptimizationOptions<float> opts;
        trust_newton(x, f<arma::fvec>, J<float>, opts);
        if (arma::norm(f(x)) > opts.ftol) {
            std::cout << "trust_newton failed armadillo single precision w/ jacobian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed\n";

    return 0;
}