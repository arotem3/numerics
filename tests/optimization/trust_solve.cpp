#include <iostream>

#include <armadillo>
#include "numerics/optimization/trust_solve.hpp"

template <class vec, typename real=typename vec::value_type>
vec f(const vec& v)
{
    real x = v[0], y = v[1];
    vec fv(2);
    fv[0] = x*x + y*y - 1;
    fv[1] = x*x*x*y - 5*y*y*x - 1;
    return fv;
}

template <typename real>
arma::Mat<real> J(const arma::Col<real>& v)
{
    real x = v[0], y = v[1];
    arma::Mat<real> J = {
        {2*x, 2*y},
        {3*x*x*y - 5*y*y, x*x*x - 10*x*y}
    };

    return J;
}

using numerics::optimization::trust_solve;
using numerics::optimization::OptimizationOptions;

int main()
{
    int n_passed = 0;
    int n_failed = 0;

    { // test 1: double precision arma no jacobian
        arma::vec x = {-2.0,2.0};
        double f0 = arma::norm(f(x));

        OptimizationOptions<double> opts;
        trust_solve(x, f<arma::vec>, opts);
        
        if (arma::norm(f(x)) > opts.ftol * f0) {
            std::cout << "trust_solve failed armadillo double precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 2: single precision arma no jacobian
        arma::fvec x = {-2.0f, 2.0f};
        float f0 = arma::norm(f(x));
        
        OptimizationOptions<float> opts;
        trust_solve(x, f<arma::fvec>, opts);
        
        if (arma::norm(f(x)) > opts.ftol * f0) {
            std::cout << "trust_solve failed armadillo single precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 3: double precision arma w/ jacobian
        arma::vec x = {-2.0, 2.0};
        double f0 = arma::norm(f(x));
        
        OptimizationOptions<double> opts;
        trust_solve(x, f<arma::vec>, J<double>, opts);

        if (arma::norm(f(x)) > opts.ftol * f0) {
            std::cout << "trust_solve failed armadillo double precision w/ jacobian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 4: single precision arma w/ jacobian
        arma::fvec x = {-2.0f, 2.0f};
        float f0 = arma::norm(f(x));
        
        OptimizationOptions<float> opts;
        trust_solve(x, f<arma::fvec>, J<float>, opts);

        if (arma::norm(f(x)) > opts.ftol * f0) {
            std::cout << "trust_solve failed armadillo single precision w/ jacobian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed\n";

    return 0;
}