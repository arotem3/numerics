#include <iostream>

#include <armadillo>
#include "numerics/optimization/trust_solve.hpp"

template <class vec, typename scalar=typename vec::value_type>
vec f(const vec& v)
{
    scalar x = v[0], y = v[1];
    vec fv(2);
    fv[0] = x*x + y*y - scalar(1.0);
    fv[1] = x*x*x*y - scalar(5.0)*y*y*x - scalar(1.0);
    return fv;
}

template <typename scalar>
arma::Mat<scalar> J(const arma::Col<scalar>& v)
{
    scalar x = v[0], y = v[1];
    arma::Mat<scalar> J = {
        {scalar(2.0)*x, scalar(2.0)*y},
        {scalar(3.0)*x*x*y - scalar(5.0)*y*y, x*x*x - scalar(10.0)*x*y}
    };

    return J;
}

using numerics::optimization::trust_solve;
using numerics::optimization::TrustOptions;
using namespace std::complex_literals;

int main()
{
    int n_passed = 0;
    int n_failed = 0;

    { // double precision arma no jacobian
        arma::vec x = {-2.0,2.0};
        double f0 = arma::norm(f(x));

        TrustOptions<double> opts;
        trust_solve(x, f<arma::vec>, opts);
        
        if (arma::norm(f(x)) > opts.ftol * f0) {
            std::cout << "trust_solve failed armadillo double precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // single precision arma no jacobian
        arma::fvec x = {-2.0f, 2.0f};
        float f0 = arma::norm(f(x));
        
        TrustOptions<float> opts;
        trust_solve(x, f<arma::fvec>, opts);
        
        if (arma::norm(f(x)) > opts.ftol * f0) {
            std::cout << "trust_solve failed armadillo single precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // complex double precision no jacobian
        arma::cx_vec x = {-2.0+1.0i,2.0-1.0i};
        double f0 = arma::norm(f(x));

        TrustOptions<double> opts;
        trust_solve(x, f<arma::cx_vec>, opts);
        
        if (arma::norm(f(x)) > opts.ftol * f0) {
            std::cout << "trust_solve failed complex double precision test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // double precision arma w/ jacobian
        arma::vec x = {-2.0, 2.0};
        double f0 = arma::norm(f(x));
        
        TrustOptions<double> opts;
        trust_solve(x, f<arma::vec>, J<double>, opts);

        if (arma::norm(f(x)) > opts.ftol * f0) {
            std::cout << "trust_solve failed armadillo double precision w/ jacobian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // single precision arma w/ jacobian
        arma::fvec x = {-2.0f, 2.0f};
        float f0 = arma::norm(f(x));
        
        TrustOptions<float> opts;
        trust_solve(x, f<arma::fvec>, J<float>, opts);

        if (arma::norm(f(x)) > opts.ftol * f0) {
            std::cout << "trust_solve failed armadillo single precision w/ jacobian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // complex double precision w/ jacobian
        arma::cx_vec x = {-2.0+1.0i, 2.0-1.0i};
        double f0 = arma::norm(f(x));
        
        TrustOptions<double> opts;
        trust_solve(x, f<arma::cx_vec>, J<arma::cx_double>, opts);
        if (arma::norm(f(x)) > opts.ftol * f0) {
            std::cout << "trust_solve failed complex double precision w/ jacobian test\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed\n";

    return 0;
}