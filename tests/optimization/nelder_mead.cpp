#include <iostream>
#include <cmath>
#include <armadillo>
#include <valarray>
#include "numerics/optimization/nelder_mead.hpp"

using numerics::optimization::nelder_mead;
using numerics::optimization::NelderMeadOptions;
using numerics::optimization::ExitFlag;

// this test verifies nelder_mead() by attempting to minimize a non-convex function
// which has one global min, and one local max. We verify that the minimization
// is performed correctly by starting near the local maximum.

template <typename real>
real g(real x)
{
    return std::exp(-x*x);
}

template <class vec, typename real = typename vec::value_type>
real f(const vec& v)
{
    static const real x0 = -1, y0 = -1, x1 = 1, y1 = 1, tau = 0.05;
    
    real x = v[0];
    real y = v[1];
    
    real f0 = g(x-x0)*g(y-y0);
    real f1 = g(x-x1)*g(y-y1);
    real r = x*x + y*y;

    return f0 - f1 + tau * r;
}

int main()
{
    int n_passed = 0;
    int n_failed = 0;

    { // test 1: arma, double, no hess
        arma::vec x =  {-0.9,-1.1};
        NelderMeadOptions<double> opts;
        opts.initial_radius = 0.5; // need to initialize this value to some sensible initial search radius for the problem
        auto rslts = nelder_mead(x, f<arma::vec>, opts);

        if (rslts.flag != ExitFlag::CONVERGED) {
            std::cout << "nelder_mead() failed armadillo double precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 2: arma, single, no hess
        arma::fvec x =  {-0.9f,-1.1f};
        NelderMeadOptions<float> opts;
        opts.initial_radius = 0.5;
        auto rslts = nelder_mead(x, f<arma::fvec>, opts);

        if (rslts.flag != ExitFlag::CONVERGED) {
            std::cout << "nelder_mead() failed armadillo single precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 3: valarray, double
        std::valarray<double> x =  {-0.9,-1.1};
        NelderMeadOptions<double> opts;
        opts.initial_radius = 0.5;
        auto rslts = nelder_mead(x, f<std::valarray<double>>, opts);

        if (rslts.flag != ExitFlag::CONVERGED) {
            std::cout << "nelder_mead() failed valarray double precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 4: valarray, float, non-const function
        struct Func
        {
            int n_evals;

            Func() : n_evals(0) {}

            float operator()(const std::valarray<float>& x)
            {
                ++n_evals;
                return f(x);
            }
        };

        Func foo;
        std::valarray<float> x =  {-0.9f,-1.1f};
        NelderMeadOptions<float> opts;
        opts.initial_radius = 0.5;
        auto rslts = nelder_mead(x, std::ref(foo), opts);

        if ((rslts.flag != ExitFlag::CONVERGED) and (foo.n_evals > 0)) {
            std::cout << "nelder_mead() failed valarray float precision w/ referenced object test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed\n";

    return 0;
}