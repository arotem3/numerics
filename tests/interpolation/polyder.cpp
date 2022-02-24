#include <iostream>
#include <complex>

#include "numerics/interpolation/polyder.hpp"

using numerics::polyder;

bool TEST_polyder_double()
{
    double p[] = {1./3., 0.5, 1., 0.};

    double dp1[] = {1., 1., 1.};
    double dp2[] = {2., 1.};

    double Dp1[3];
    polyder<double>(p, p+4, Dp1);

    double Dp2[2];
    polyder<double>(p, p+4, Dp2, 2);

    bool success = true;
    for (int i=0; i < 3; ++i)
        success = (std::abs(Dp1[i] - dp1[i]) < 1e-12) && success;

    for (int i=0; i < 2; ++i)
        success = (std::abs(Dp2[i] - dp2[i]) < 1e-12) && success;

    if (not success)
        std::cout << "polyder() failed double test\n";

    return success;
}

bool TEST_polyder_complex()
{
    std::complex<float> p[] = {1./3.f, 0.5f, 1.f, 0.f};

    std::complex<float> dp1[] = {1.f, 1.f, 1.f};
    std::complex<float> dp2[] = {2.f, 1.f};

    std::complex<float> Dp1[3];
    polyder<float>(p, p+4, Dp1);

    std::complex<float> Dp2[2];
    polyder<float>(p, p+4, Dp2, 2);

    bool success = true;
    for (int i=0; i < 3; ++i)
        success = (std::abs(Dp1[i] - dp1[i]) < 1e-6f) && success;

    for (int i=0; i < 2; ++i)
        success = (std::abs(Dp2[i] - dp2[i]) < 1e-6f) && success;

    if (not success)
        std::cout << "polyder() failed complex test\n";

    return success;
}

int main()
{
    int n_success = 0;
    int n = 0;

    n_success += TEST_polyder_double(); ++n;
    n_success += TEST_polyder_complex(); ++n;

    std::cout << "polyder succeeded " << n_success << " / " << n << " tests.\n";

    return 0;
}