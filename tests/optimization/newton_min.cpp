#include <iostream>

#include <cmath>
#include <armadillo>
#include <valarray>
#include "numerics/optimization/newton_min.hpp"

using numerics::optimization::newton_min;
using numerics::optimization::OptimizationOptions;

// this test verifies newton_min() by attempting to minimize a non-convex
// function which has one global min, one local max, and one saddle point. We
// verify that the minimization is performed correctly by starting near the
// local maximum.

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
        arma::vec x = {1.1, 0.1};
        OptimizationOptions<double> opts;
        newton_min(x, f<arma::vec>, df<arma::vec>, opts);

        bool first_order = arma::norm(df(x)) < opts.ftol; // check gradient = 0
        bool second_order = arma::all(arma::eig_sym(H(x)) > 0); // check hessian > 0

        if (not first_order and not second_order) {
            std::cout << "newton_min() failed armadillo double precision hessian free test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 2: arma, single, no hess
        arma::fvec x = {1.1f, 0.1f};
        OptimizationOptions<float> opts;
        newton_min(x, f<arma::fvec>, df<arma::fvec>, opts);

        bool first_order = arma::norm(df(x)) < opts.ftol;
        bool second_order = arma::all(arma::eig_sym(H(x)) > 0);

        if (not (first_order and second_order)) {
            std::cout << "newton_min() failed armadillo single precision hessian free test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 3: arma, double, hess
        arma::vec x = {1.1, 0.1};
        OptimizationOptions<double> opts;
        newton_min(x, f<arma::vec>, df<arma::vec>, H<double>, opts);

        bool first_order = arma::norm(df(x)) < opts.ftol;
        bool second_order = arma::all(arma::eig_sym(H(x)) > 0);

        if (not (first_order and second_order)) {
            std::cout << "newton_min() failed armadillo double precision w/ hessian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 4: arma, float, hess
        arma::fvec x = {1.1f, 0.1f};
        OptimizationOptions<float> opts;
        newton_min(x, f<arma::fvec>, df<arma::fvec>, H<float>, opts);

        bool first_order = arma::norm(df(x)) < opts.ftol;
        bool second_order = arma::all(arma::eig_sym(H(x)) > 0);

        if (not (first_order and second_order)) {
            std::cout << "newton_min() failed armadillo single precision w/ hessian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 5: valarray, double
        std::valarray<double> x = {1.1, 0.1};
        OptimizationOptions<double> opts;
        newton_min(x, f<std::valarray<double>>, df<std::valarray<double>>, opts);

        arma::vec z(2); z[0] = x[0]; z[1] = x[1];

        bool first_order = arma::norm(df(z)) < opts.ftol; // check gradient = 0
        bool second_order = arma::all(arma::eig_sym(H(z)) > 0); // check hessian > 0

        if (not (first_order and second_order)) {
            std::cout << "newton_min() failed valarray double precision hessian free test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 6: valarray, float, non-const function
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
        std::valarray<float> x = {1.1f, 0.1f};
        OptimizationOptions<float> opts;
        newton_min(x, f<std::valarray<float>>, std::ref(g), opts);

        arma::vec z(2); z[0] = x[0]; z[1] = x[1];

        bool first_order = arma::norm(df(z)) < opts.ftol; // check gradient = 0
        bool second_order = arma::all(arma::eig_sym(H(z)) > 0); // check hessian > 0

        if (not (first_order and second_order and (g.n_evals > 0))) {
            std::cout << "newton_min() failed valarray float precision hessian free test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 7: arma, double, sparse hessian
        int n = 10;
        arma::sp_mat A = 2*arma::speye(n,n);
        A.diag(-1).fill(-1);
        A.diag(1).fill(1);

        auto vfun = [](const arma::vec& x) -> arma::vec
        {
            return arma::exp(-arma::square(x));
        };

        auto dfun = [&](const arma::vec& x) -> arma::vec
        {
            return -2*x%vfun(x);
        };

        auto d2fun = [&](const arma::vec& x) -> arma::vec
        {
            return (4*x%x - 2)%vfun(x);
        };

        auto fun = [&](const arma::vec& x) -> double
        {
            return arma::sum(vfun(A*x - 1) - vfun(A*x + 1)) + 0.05*arma::dot(x,x);
        };

        auto grad = [&](const arma::vec& x) -> arma::vec
        {
            return A.t()*(dfun(A*x-1) - dfun(A*x+1)) + 0.1*x;
        };

        auto hess = [&](const arma::vec& x) -> arma::sp_mat
        {
            arma::sp_mat D(n,n);
            D.diag() = d2fun(A*x - 1) - d2fun(A*x + 1);
            return A.t()*D*A + 0.1*arma::speye(n,n);
        };
    
        arma::vec x = 0.5*arma::ones(n); // this produces a negative definite hessian
        OptimizationOptions<double> opts;
        newton_min(x, fun, grad, hess, opts);

        bool first_order = arma::norm(grad(x)) < opts.ftol;
        bool second_order = arma::all(arma::eig_sym(arma::mat(hess(x))) > 0);

        if (not (first_order and second_order)) {
            std::cout << "newton_min() failed armadillo double precision w/ sparse hessian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 8: arma, float, sparse hessian product
        int n = 10;
        arma::sp_fmat A = 2*arma::speye(n,n);
        A.diag(-1).fill(-1);
        A.diag(1).fill(1);

        auto vfun = [](const arma::fvec& x) -> arma::fvec
        {
            return arma::exp(-arma::square(x));
        };

        auto dfun = [&](const arma::fvec& x) -> arma::fvec
        {
            return -2*x%vfun(x);
        };

        auto d2fun = [&](const arma::fvec& x) -> arma::fvec
        {
            return (4*x%x - 2)%vfun(x);
        };

        auto fun = [&](const arma::fvec& x) -> float
        {
            return arma::sum(vfun(A*x - 1) - vfun(A*x + 1)) + 0.05f*arma::dot(x,x);
        };

        auto grad = [&](const arma::fvec& x) -> arma::fvec
        {
            return A.t()*(dfun(A*x-1) - dfun(A*x+1)) + 0.1f*x;
        };

        auto hess = [&](const arma::fvec& x, const arma::fvec& v) -> arma::fvec
        {
            arma::fvec D = d2fun(A*x - 1) - d2fun(A*x + 1);
            return A.t()*(D%(A*v)) + 0.1f*v;
        };

        arma::fvec x = 0.5*arma::ones<arma::fvec>(n);
        OptimizationOptions<float> opts;
        newton_min(x, fun, grad, hess, opts);

        bool first_order = arma::norm(grad(x)) < opts.ftol;

        if (not first_order) {
            std::cout << "newton_min() failed armadillo single precision w/ hessian operator test\n";
            ++n_failed;
        }
        else
            ++n_passed;

    }
    
    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed\n";

    return 0;
}