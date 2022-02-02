#include <iostream>

#include <armadillo>
#include <valarray>
#include "numerics/optimization/newton.hpp"

template <class vec, typename scalar=typename vec::value_type>
vec f(const vec& v)
{
    scalar x = v[0], y = v[1];
    vec fv(2);
    fv[0] = x*x + y*y - scalar(1.0);
    fv[1] = x + y - scalar(std::sqrt(2.0));
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

using numerics::optimization::newton;
using numerics::optimization::OptimizationOptions;
using namespace std::complex_literals;

int main()
{
    int n_passed = 0;
    int n_failed = 0;

    { // double precision arma no jacobian
        arma::vec x = {-0.01, 0.0}; // jac is singular at 0,0 so start off 0,0
        double f0 = std::max(1.0, arma::norm(f(x)));

        OptimizationOptions<double> opts;
        newton(x, f<arma::vec>, opts);
        
        if (arma::norm(f(x)) > opts.ftol * f0) {
            std::cout << "newton failed armadillo double precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // single precision arma no jacobian
        arma::fvec x = {-0.01f, 0.0f}; // jac is singular at 0,0 so start off 0,0
        float f0 = std::max(1.0f, arma::norm(f(x)));
        
        OptimizationOptions<float> opts;
        newton(x, f<arma::fvec>, opts);
        if (arma::norm(f(x)) > opts.ftol * f0) {
            std::cout << "newton failed armadillo single precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // complex double precision arma no jacobian
        arma::cx_vec x = {-0.01+0.01i, 0.0-0.01i}; // jac is singular at 0,0 so start off 0,0
        double f0 = std::max(1.0, arma::norm(f(x)));

        OptimizationOptions<double> opts;
        newton(x, f<arma::cx_vec>, opts);
        
        if (arma::norm(f(x)) > opts.ftol * f0) {
            std::cout << "newton failed armadillo complex double precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // double precision arma w/ jacobian
        arma::vec x = {-0.01, 0.0}; // jac is singular at 0,0 so start off 0,0
        double f0 = std::max(1.0, arma::norm(f(x)));
        
        OptimizationOptions<double> opts;
        newton(x, f<arma::vec>, J<double>, opts);
        if (arma::norm(f(x)) > opts.ftol * f0) {
            std::cout << "newton failed armadillo double precision w/ jacobian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // single precision arma w/ jacobian
        arma::fvec x = {-0.01f, 0.0f}; // jac is singular at 0,0 so start off 0,0
        float f0 = std::max(1.0f, arma::norm(f(x)));
        
        OptimizationOptions<float> opts;
        newton(x, f<arma::fvec>, J<float>, opts);
        if (arma::norm(f(x)) > opts.ftol * f0) {
            std::cout << "newton failed armadillo single precision w/ jacobian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // complex double precision arma w/ jacobian
        arma::cx_vec x = {-0.01+0.01i, 0.0-0.01i}; // jac is singular at 0,0 so start off 0,0
        double f0 = std::max(1.0, arma::norm(f(x)));

        OptimizationOptions<double> opts;
        newton(x, f<arma::cx_vec>, J<arma::cx_double>, opts);
        
        if (arma::norm(f(x)) > opts.ftol * f0) {
            std::cout << "newton failed armadillo complex double precision w/ jacobian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // double precision valarray
        std::valarray<double> x = {-0.01, 0.0};
        double f0 = std::max(1.0, std::sqrt((f(x)*f(x)).sum()));
        
        OptimizationOptions<double> opts;
        newton(x, f<std::valarray<double>>, opts);
        if ( (f(x) * f(x)).sum() > opts.ftol * f0) {
            std::cout << "newton failed valarray double precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // complex single precision valarray
        std::valarray<std::complex<float>> x = {-0.01f+0.01if, 0.0f-0.01if};
        float f0 = std::max(1.0f, numerics::__vmath::norm_impl(f(x)));

        OptimizationOptions<float> opts;
        newton(x, f<std::valarray<std::complex<float>>>, opts);
        if (numerics::__vmath::norm_impl(f(x)) > opts.ftol * f0) {
            std::cout << "newton failed valarray complex single precision test\n";
            ++n_failed;
        } else
            ++n_passed;
    }

    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed\n";

    return 0;
}