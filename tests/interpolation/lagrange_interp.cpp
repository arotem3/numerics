#include "numerics/interpolation/lagrange_interp.hpp"
#include <cmath>
#include <complex>
#include <valarray>
#include <vector>
#include <iostream>

using namespace std::complex_literals;
using numerics::lagrange_interp;

bool TEST_lagrange_interp_double()
{
    auto g = [](double x) -> double
    {
        return x*(x*x - 1.);
    };

    double x[] = {0.0, 0.25, 0.75, 1.0};
    double y[4];
    for (int i=0; i < 4; ++i)
        y[i] = g(x[i]);

    double xx = 0.5;
    double yy = 0;
    lagrange_interp(x, x+4, y, xx, yy);

    bool success = std::abs(yy - g(xx)) < 1e-12;

    if (not success)
        std::cout << "lagrange_interp() failed double test\n";

    return success;
}

bool TEST_lagrange_interp_valarray()
{
    auto g = [](float x) -> std::valarray<float>
    {
        std::valarray<float> out(0.0, 2);
        out[0] = x*(x*x - 1);
        out[1] = std::hermite(4, x);
        return out;
    };

    std::vector<float> x = {-2.f, -1.f, 1.f, 2.f, 3.f};
    std::vector<std::valarray<float>> y(5);
    for (int i=0; i < 5; ++i)
        y[i] = g(x[i]);

    float xx = 0.0;
    std::valarray<float> yy(0.0, 2);
    lagrange_interp(x.begin(), x.end(), y.begin(), xx, yy);

    bool success = std::abs(yy - g(xx)).max() < 1e-12;

    if (not success)
        std::cout << "lagrange_interp() failed valarray test\n";

    return success;
}

bool TEST_lagrange_interp_complex()
{
    auto g = [](double x) -> std::complex<double>
    {
        std::complex<double> z = x * std::sqrt(1i);
        return z*(z*z - 1.);
    };

    double x[] = {0., 0.25, 0.75, 1.0};
    std::vector<std::complex<double>> y(4);
    for (int i=0; i < 4; ++i)
        y[i] = g(x[i]);
    
    double xx = 0.5;
    std::complex<double> yy = 0.0;

    lagrange_interp(x, x+4, y.begin(), xx, yy);

    bool success = std::abs(yy - g(xx)) < 1e-12;

    if (not success)
        std::cout << "lagrange_interp() failed complex test\n";

    return success;
}

int main()
{
    int n_success = 0;
    int n = 0;

    n_success += TEST_lagrange_interp_double(); ++n;
    n_success += TEST_lagrange_interp_valarray(); ++n;
    n_success += TEST_lagrange_interp_complex(); ++n;

    std::cout << "lagrange_interp succeeded " << n_success << " / " << n << " tests.\n";

    return 0;
}