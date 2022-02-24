#include <iostream>
#include <valarray>
#include <complex>

#include "numerics/interpolation/CollocPoly.hpp"

using namespace std::complex_literals;

// verify that interpolation is correct on polynomial data
bool TEST_colloc_poly_double()
{
    auto g = [](double z) -> double
    {
        return z*(z*z - 1);
    };
    
    double x[] = {0.0, 0.25, 0.75, 1.0};
    double f[4];
    for (int i=0; i < 4; ++i)
        f[i] = g(x[i]);

    numerics::CollocPoly<double,double> p(x, x+4, f, f+4);

    bool success = std::abs(g(0.5) - p(0.5)) < 1e-12;
    if (not success)
        std::cout << "CollocPoly failed interpolation test for vec = double.\n";
    
    return success;
}

bool TEST_colloc_poly_complex()
{
    auto g = [](double z) -> std::complex<double>
    {
        std::complex<double> Z = z * std::sqrt(1.0i);
        return Z*(Z*Z - 1.0);
    };
    
    double x[] = {0.0, 0.25, 0.75, 1.0};
    std::complex<double> f[4];
    for (int i=0; i < 4; ++i)
        f[i] = g(x[i]);

    numerics::CollocPoly<double,std::complex<double>> p(x, x+4, f, f+4);

    bool success = std::abs(g(0.5) - p(0.5)) < 1e-12;
    if (not success)
        std::cout << "CollocPoly failed interpolation test for vec = double.\n";
    
    return success;
}

bool TEST_cheb_float()
{
    auto g = [](float z) -> float
    {
        return z*(z*z - 1);
    };

    numerics::ChebInterp<float,float> p(4, 1.f, 2.5f, g);

    bool success = std::abs(g(0.5f) - p(0.5f)) < 1e-6;
    if (not success)
        std::cout << "ChebInterp failed interpolation test for vec = float.\n";
    
    return success;
}

bool TEST_cheb_valarray()
{
    auto g = [](double z) -> std::valarray<double>
    {
        std::valarray<double> out(0.0, 2);
        out[0] = z*(z*z - 1);
        out[1] = std::hermite(4, z);
        return out;
    };

    numerics::ChebInterp<double,std::valarray<double>> p(5, -1.1, 2.0, g);

    bool success = std::abs(g(0.0) - p(0.0)).max() < 1e-12;
    if (not success)
        std::cout << "ChebInterp failed interpolation test for vec = std::valarray<double>.\n";
    return success;
}

int main()
{
    int n_success = 0;
    int n = 0;

    n_success += TEST_colloc_poly_double(); ++n;
    n_success += TEST_colloc_poly_complex(); ++n;
    n_success += TEST_cheb_float(); ++n;
    n_success += TEST_cheb_valarray(); ++n;

    std::cout << "CollocPoly succeeded " << n_success << " / " << n << " tests.\n";

    return 0;
}