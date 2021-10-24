#include <iostream>

#include <armadillo>
#include "numerics/optimization/mixFPI.hpp"

using numerics::optimization::mixFPI;
using numerics::optimization::OptimizationOptions;

template <typename real>
arma::Col<real> f(const arma::Col<real>& v)
{
    real x = v[0], y = v[1];
    arma::Col<real> fv(2);
    fv[0] = x*y - 1;
    fv[1] = 0.6*y;
    return fv;
}

int main()
{
    int n_passed = 0;
    int n_failed = 0;

    { // test 1: double
        arma::vec x = {5.0,3.0};

        OptimizationOptions<double> opts;
        mixFPI<5>(x, f<double>, opts);
        
        if (arma::norm(x - f(x)) > opts.ftol)
        {
            std::cout << "mixFPI double precision test failed" << std::endl;
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 2: float
        arma::fvec x = {5.0f,3.0f};

        OptimizationOptions<float> opts;
        mixFPI<5>(x, f<float>, opts);
        
        if (arma::norm(x - f(x)) > opts.ftol)
        {
            std::cout << "mixFPI double precision test failed" << std::endl;
            ++n_failed;
        }
        else
            ++n_passed;
    }

    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed\n";

    return 0;
}