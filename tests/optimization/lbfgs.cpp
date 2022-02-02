#include <iostream>
#include <cmath>
#include <armadillo>
#include <valarray>
#include "numerics/optimization/lbfgs.hpp"

using numerics::optimization::lbfgs;
using numerics::optimization::LBFGS_Options;

// this test verifies lbfgs() by attempting to minimize a non-convex function
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

template <class vec, typename real = typename vec::value_type>
vec df(const vec& v)
{
    static const real x0 = -1, y0 = -1, x1 = 1, y1 = 1, tau = 0.05;
    
    real x = v[0];
    real y = v[1];
    
    real f0 = g(x-x0)*g(y-y0);
    real f1 = g(x-x1)*g(y-y1);
    real r = x*x + y*y;
    
    vec g(2);
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
        LBFGS_Options<double> opts;
        lbfgs(x, f<arma::vec>, df<arma::vec>, opts);

        bool first_order = arma::norm(df(x)) < opts.ftol; // check gradient = 0
        bool second_order = arma::all(arma::eig_sym(H(x)) > 0); // check hessian > 0

        if (not first_order and not second_order) {
            std::cout << "lbfgs() failed armadillo double precision hessian free test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 2: arma, single, no hess
        arma::fvec x =  {-0.9f,-1.1f};
        LBFGS_Options<float> opts;
        lbfgs(x, f<arma::fvec>, df<arma::fvec>, opts);

        bool first_order = arma::norm(df(x)) < opts.ftol;
        bool second_order = arma::all(arma::eig_sym(H(x)) > 0);

        if (not (first_order and second_order)) {
            std::cout << "lbfgs() failed armadillo single precision hessian free test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 3: valarray, double
        std::valarray<double> x =  {-0.9,-1.1};
        LBFGS_Options<double> opts;
        lbfgs(x, f<std::valarray<double>>, df<std::valarray<double>>, opts);

        arma::vec z(2); z[0] = x[0]; z[1] = x[1];

        bool first_order = arma::norm(df(z)) < opts.ftol; // check gradient = 0
        bool second_order = arma::all(arma::eig_sym(H(z)) > 0); // check hessian > 0

        if (not (first_order and second_order)) {
            std::cout << "lbfgs() failed valarray double precision hessian free test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 4: valarray, float, non-const function
        struct Grad
        {
            int n_evals;

            Grad() : n_evals(0) {}

            std::valarray<float> operator()(const std::valarray<float>& x)
            {
                ++n_evals;
                return df(x);
            }
        };

        Grad g;
        std::valarray<float> x =  {-0.9f,-1.1f};
        LBFGS_Options<float> opts;
        lbfgs(x, f<std::valarray<float>>, std::ref(g), opts);

        arma::vec z(2); z[0] = x[0]; z[1] = x[1];

        bool first_order = arma::norm(df(z)) < opts.ftol; // check gradient = 0
        bool second_order = arma::all(arma::eig_sym(H(z)) > 0); // check hessian > 0

        if (not (first_order and second_order and (g.n_evals > 0))) {
            std::cout << "lbfgs() failed valarray float precision hessian free test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed\n";

    return 0;
}