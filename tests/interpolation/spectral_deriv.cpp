#include <iostream>
#include <valarray>
#include <armadillo>

#include "numerics/interpolation/spectral_deriv.hpp"
using numerics::spectral_deriv;
using numerics::ChebInterp;

int TEST_spectral_deriv_double()
{
    auto f = [](double x) -> double 
    {
        return std::exp(std::sin(M_PI*x) - x*x);
    };

    auto df = [&f](double x) -> double
    {
        return (M_PI*std::cos(M_PI*x) - 2*x) * f(x);
    };

    ChebInterp<double,double> p = spectral_deriv(f, -2., 2., 128);

    double max_err = 0;
    for (int i=0; i < 128; ++i)
    {
        double err = std::abs(p.f[i] - df(p.x[i]));
        max_err = std::max(err, max_err);
    }

    bool success = max_err < 1e-10;

    if (not success)
        std::cout << "spectral_deriv() failed double test.\n";

    return success;
}

int TEST_spectral_deriv_valarray()
{
    auto f = [](double x) -> std::valarray<double>
    {
        std::valarray<double> y(2);
        y[0] = std::exp(-16.*x*x);
        y[1] = std::sin(20.*x*x);

        return y;
    };

    auto df = [](double x) -> std::valarray<double>
    {
        std::valarray<double> y(2);
        y[0] = -32.*x*std::exp(-16.*x*x);
        y[1] = 40.*x*std::cos(20*x*x);

        return y;
    };

    auto p = spectral_deriv<std::valarray<double>>(f, 0., 1., 128);

    double max_err = 0;
    for (int i=0; i < 128; ++i)
    {
        double err = std::abs(p.f[i] - df(p.x[i])).max();
        max_err = std::max(err, max_err);
    }

    bool success = max_err < 1e-10;
    if (not success)
        std::cout << "spectral_deriv() failed valarray test.\n";
    
    return success;
}

int TEST_spectral_deriv_arma()
{
    auto f = [](double x) -> arma::vec
    {
        arma::vec y(2);
        y[0] = std::exp(-16.*x*x);
        y[1] = std::sin(20.*x*x);

        return y;
    };

    auto df = [](double x) -> arma::vec
    {
        arma::vec y(2);
        y[0] = -32.*x*std::exp(-16.*x*x);
        y[1] = 40.*x*std::cos(20*x*x);

        return y;
    };

    auto p = spectral_deriv<arma::vec>(f, 0., 1., 128);

    double max_err = 0;
    for (int i=0; i < 128; ++i)
    {
        double err = arma::abs(p.f[i] - df(p.x[i])).max();
        max_err = std::max(err, max_err);
    }

    bool success = max_err < 1e-10;
    if (not success)
        std::cout << "spectral_deriv() failed valarray test.\n";
    
    return success;
}

int main()
{
    int n_success = 0;
    int n = 0;

    n_success += TEST_spectral_deriv_double(); ++n;
    n_success += TEST_spectral_deriv_valarray(); ++n;
    n_success += TEST_spectral_deriv_arma(); ++n;

    std::cout << "spectral_deriv succeeded " << n_success << " / " << n << " tests.\n";

    return 0;
}