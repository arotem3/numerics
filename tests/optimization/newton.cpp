#include <iostream>

#include <armadillo>
#include <valarray>
#include "numerics/optimization/newton.hpp"

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

using numerics::optimization::newton;
using numerics::optimization::OptimizationOptions;

int main()
{
    int n_passed = 0;
    int n_failed = 0;

    { // test 1: double precision arma no jacobian
        arma::vec x = {-0.01, 0.0}; // jac is singular at 0,0 so start off 0,0
        
        OptimizationOptions<double> opts;
        newton(x, f<arma::vec>, opts);
        if (arma::norm(f(x)) > opts.ftol) {
            std::cout << "newton failed armadillo double precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 2: single precision arma no jacobian
        arma::fvec x = {-0.01f, 0.0f}; // jac is singular at 0,0 so start off 0,0
        
        OptimizationOptions<float> opts;
        newton(x, f<arma::fvec>, opts);
        if (arma::norm(f(x)) > opts.ftol) {
            std::cout << "newton failed armadillo single precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 3: double precision arma w/ jacobian
        arma::vec x = {-0.01, 0.0}; // jac is singular at 0,0 so start off 0,0
        
        OptimizationOptions<double> opts;
        newton(x, f<arma::vec>, J<double>, opts);
        if (arma::norm(f(x)) > opts.ftol) {
            std::cout << "newton failed armadillo double precision w/ jacobian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 4: single precision arma w/ jacobian
        arma::fvec x = {-0.01f, 0.0f}; // jac is singular at 0,0 so start off 0,0
        
        OptimizationOptions<float> opts;
        newton(x, f<arma::fvec>, J<float>, opts);
        if (arma::norm(f(x)) > opts.ftol) {
            std::cout << "newton failed armadillo single precision w/ jacobian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 5: double precision valarray
        std::valarray<double> x = {-0.01, 0.0};
        
        OptimizationOptions<double> opts;
        newton(x, f<std::valarray<double>>, opts);
        if ( (f(x) * f(x)).sum() > opts.ftol ) {
            std::cout << "newton failed valarray double precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed\n";

    return 0;
}