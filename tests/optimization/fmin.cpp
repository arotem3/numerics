#include <iostream>

#include "numerics/optimization/fmin.hpp"

using namespace numerics::optimization;

// piecewise quadratic function with smooth minimum at x=0.1
template <typename real>
real f(real x)
{
    if (x < 0.1)
        return 1 - 10*x + 50*x*x;
    else
        return 0.5 + (50.0/27.0)*std::pow(x-0.1,2);
}

int main()
{
    int n_passed = 0;
    int n_failed = 0;

    { // test 1: double prec. fminbnd
        double x = fminbnd<double>(f<double>, 0.0, 2.0, 1e-8);
        if (std::abs(x - 0.1) > 1e-8) {
            std::cout << "double precision fminbnd test failed.\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 2: single prec. fminbnd
        float x = fminbnd<float>(f<float>, 0.0f, 2.0f, 1e-4f);
        if (std::abs(x - 0.1f) > 1e-4f) {
            std::cout << "single precision fminbnd test failed.\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 3: double prec. fminsearch
        double x = fminsearch<double>(f<double>, 0.0, 1e-5, 1e-8);
        if (std::abs(x - 0.1) > 1e-8) {
            std::cout << "double precision fminsearch test failed.\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    { // test 4: single prec. fminsearch
        double x = fminsearch<float>(f<double>, 0.0, 1e-2f, 1e-4f);
        if (std::abs(x - 0.1f) > 1e-4f) {
            std::cout << "double precision fminsearch test failed.\n";
            ++n_failed;
        }
        else
            ++n_passed;
    }

    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed\n";

    return 0;
}