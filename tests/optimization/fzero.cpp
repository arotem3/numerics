#include <iostream>

#include "numerics/optimization/fzero.hpp"

using namespace numerics::optimization;
using namespace std::complex_literals;

template <typename real>
real f(real x)
{
    return std::exp(x) - x*x;
}

template <typename real>
real df(real x)
{
    return std::exp(x) - 2*x;
}

template <std::floating_point T>
std::complex<T> g(std::complex<T> z)
{
    return z*z*z - T(1);
}

template <std::floating_point T>
std::complex<T> dg(std::complex<T> z)
{
    return T(3)*z*z;
}

int main()
{
    int n_passed = 0;
    int n_failed = 0;

    { // double prec. fzero
        double x = fzero<double>(f<double>, -3.0, 3.0, 1e-8);
        if (std::abs(f(x)) > 1e-8) {
            std::cout << "double precision fzero test failed.\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // single prec. fzero
        float x = fzero<float>(f<float>, -3.0f, 3.0f, 1e-4f);
        if (std::abs(f(x)) > 1e-4f)
        {
            std::cout << "single precision fzero test failed.\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // double prec. newton_1d + derivative
        double x = newton_1d<double>(f<double>, df<double>, 3.0, 1e-8);
        if (std::abs(f(x)) > 1e-8) {
            std::cout << "double precision newton_1d with derivative test failed.\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // single prec. newton_1d + derivative
        float x = newton_1d<float>(f<float>, df<float>, 3.0f, 1e-4f);
        if (std::abs(f(x)) > 1e-4f)
        {
            std::cout << "single precision newton_1d with derivative test failed.\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // double prec. newton_1d, no derivative
        double x = newton_1d<double>(f<double>, 3.0, 1e-8);
        if (std::abs(f(x)) > 1e-8) {
            std::cout << "double precision newton_1d without derivative test failed.\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // single prec. newton_1d no derivative
        float x = newton_1d<float>(f<float>, 3.0f, 1e-4f);
        if (std::abs(f(x)) > 1e-4f)
        {
            std::cout << "single precision newton_1d without derivative test failed.\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }
    
    { // complex double prec. newton_1d + derivative
        std::complex<double> z = newton_1d(g<double>, dg<double>, 1.0+1.0i, 1e-8);
        if (std::abs(g(z)) > 1e-8) {
            std::cout << "complex double precision newton_1d with derivative test failed.\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // complex double prec. newton_1d no derivative
        std::complex<double> z = newton_1d(g<double>, 1.0+1.0i, 1e-8);
        if (std::abs(g(z)) > 1e-8) {
            std::cout << "complex double precision newton_1d without derivative test failed.\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // double prec. secant
        double x = secant<double>(f<double>, -3.0, 3.0, 1e-8);
        if (std::abs(f(x)) > 1e-8) {
            std::cout << "double precision secant test failed.\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // single prec. secant
        float x = secant<float>(f<float>, -3.0f, 3.0f, 1e-4f);
        if (std::abs(f(x)) > 1e-4f)
        {
            std::cout << "single precision secant test failed.\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // complex double prec. secant
        std::complex<double> z = secant(g<double>, 1.0+1.0i, 0.5-0.5i, 1e-8);
        if (std::abs(g(z)) > 1e-8) {
            std::cout << "complex double precision secant test failed.\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // double prec. bisect
        double x = bisect<double>(f<double>, -3.0, 3.0, 1e-8);
        if (std::abs(f(x)) > 1e-8) {
            std::cout << "double precision bisect test failed.\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // single prec. bisect
        float x = bisect<float>(f<float>, -3.0, 3.0, 1e-4);
        if (std::abs(f(x)) > 1e-4)
        {
            std::cout << "single precision bisect test failed.\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed\n";

    return 0;
}