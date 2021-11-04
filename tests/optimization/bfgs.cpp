#include <iostream>

#include <cmath>
#include <armadillo>
#include <valarray>
#include "numerics/optimization/bfgs.hpp"

using numerics::optimization::bfgs;
using numerics::optimization::OptimizationOptions;

// this test verifies bfgs() by attempting to minimize a non-convex function
// which has one global min, and one local max. We verify that the minimization
// is performed correctly by starting near the local maximum.

template <typename real>
real g(real x)
{
    return std::exp(-x*x);
}

template <typename real>
real f(const arma::Col<real>& v)
{
    static const real x0 = -1, y0 = -1, x1 = 1, y1 = 1, tau = 0.05;
    
    real x = v[0];
    real y = v[1];
    
    real f0 = g(x-x0)*g(y-y0);
    real f1 = g(x-x1)*g(y-y1);
    real r = x*x + y*y;

    return f0 - f1 + tau * r;
}

template <typename real>
arma::Col<real> df(const arma::Col<real>& v)
{
    static const real x0 = -1, y0 = -1, x1 = 1, y1 = 1, tau = 0.05;
    
    real x = v[0];
    real y = v[1];
    
    real f0 = g(x-x0)*g(y-y0);
    real f1 = g(x-x1)*g(y-y1);
    real r = x*x + y*y;
    
    arma::Col<real> g(2);
    g[0] = -std::hermite(1,x-x0)*f0 + std::hermite(1,x-x1)*f1 + 2*tau*x;
    g[1] = -std::hermite(1,y-y0)*f0 + std::hermite(1,y-y1)*f1 + 2*tau*y;

    return g;
}

template <typename real>
arma::Mat<real> H(const arma::Col<real>& v)
{
    static const real x0 = -1, y0 = -1, x1 = 1, y1 = 1, tau = 0.05;
    
    real x = v[0], y = v[1];
    
    real f0 = std::exp(-(x-x0)*(x-x0) - (y-y0)*(y-y0));
    real f1 = std::exp(-(x-x1)*(x-x1) - (y-y1)*(y-y1));

    arma::Mat<real> H(2,2);
    H(0,0) = std::hermite(2,x-x0)*f0 - std::hermite(2,x-x1)*f1 + 2*tau;
    H(1,1) = std::hermite(2,y-y0)*f0 - std::hermite(2,y-y1)*f1 + 2*tau;
    H(0,1) = std::hermite(1,x-x0)*std::hermite(1,y-y0)*f0 - std::hermite(1,x-x1)*std::hermite(1,y-y1)*f1;
    H(1,0) = H(0,1);

    return H;
}

int main()
{
    int n_passed = 0;
    int n_failed = 0;

    { // test 1: arma, double, no hess
        arma::vec x =  {-0.9,-1.1};
        OptimizationOptions<double> opts;
        bfgs(x, f<double>, df<double>, opts);

        bool first_order = arma::norm(df(x)) < opts.ftol; // check gradient = 0
        bool second_order = arma::all(arma::eig_sym(H(x)) > 0); // check hessian > 0

        if (not (first_order and second_order)) {
            std::cout << "bfgs() failed armadillo double precision hessian free test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 2: arma, single, no hess
        arma::fvec x =  {-0.9f,-1.1f};
        OptimizationOptions<float> opts;
        bfgs(x, f<float>, df<float>, opts);

        bool first_order = arma::norm(df(x)) < opts.ftol;
        bool second_order = arma::all(arma::eig_sym(H(x)) > 0);

        if (not (first_order and second_order)) {
            std::cout << "bfgs() failed armadillo single precision hessian free test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 3: arma, double, hess
        arma::vec x =  {-0.9,-1.1};
        OptimizationOptions<double> opts;
        bfgs(x, f<double>, df<double>, H<double>, opts);

        bool first_order = arma::norm(df(x)) < opts.ftol;
        bool second_order = arma::all(arma::eig_sym(H(x)) > 0);

        if (not (first_order and second_order)) {
            std::cout << "bfgs() failed armadillo double precision w/ hessian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 4: arma, float, hess
        arma::fvec x =  {-0.9f,-1.1f};
        OptimizationOptions<float> opts;
        bfgs(x, f<float>, df<float>, H<float>, opts);

        bool first_order = arma::norm(df(x)) < opts.ftol;
        bool second_order = arma::all(arma::eig_sym(H(x)) > 0);

        if (not (first_order and second_order)) {
            std::cout << "bfgs() failed armadillo single precision w/ hessian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }
    
    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed\n";

    return 0;
}