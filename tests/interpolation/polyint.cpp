#include <iostream>
#include <complex>

#include "numerics/interpolation/polyint.hpp"

using numerics::polyint;

bool TEST_polyint_double()
{
    double p[] = {4., 3., 2., 1.};

    double Ip[] = {1., 1., 1., 1.};

    double Ip1[4];
    polyint<double>(p, p+4, Ip1);

    bool success = true;
    for (int i=0; i < 3; ++i)
        success = (std::abs(Ip[i] - Ip1[i]) < 1e-12) && success;

    if (not success)
        std::cout << "polyint() failed double test\n";

    return success;
}

bool TEST_polyint_complex()
{
    std::complex<float> p[] = {4.f, 3.f, 2.f, 1.f};

    std::complex<float> Ip[] = {1.f, 1.f, 1.f, 1.f};

    std::complex<float> Ip1[4];
    polyint<float>(p, p+4, Ip1);

    bool success = true;
    for (int i=0; i < 3; ++i)
        success = (std::abs(Ip[i] - Ip1[i]) < 1e-6f) && success;

    if (not success)
        std::cout << "polyint() failed complex test\n";

    return success;
}

int main()
{
    int n_success = 0;
    int n = 0;

    n_success += TEST_polyint_double(); ++n;
    n_success += TEST_polyint_complex(); ++n;

    std::cout << "polyint succeeded " << n_success << " / " << n << " tests.\n";

    return 0;
}