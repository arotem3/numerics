#include "numerics.hpp"

#include <cmath>
#include <iostream>

template <typename real>
real test_func(real x) {
    return std::exp(-x*x);
}

int main() {
    long double val = 0.746824132812427025l;
    int n_passed = 0;
    int n_failed = 0;

    { // test 1
        float I = numerics::simpson_integral<float>(test_func<float>, 0.0f, 1.0f, 1e-3f);
        if (std::abs(I - val) > 1e-3) {
            std::cout << "simpson_integral failed single prec. test\n";
            ++n_failed;
        }
        else ++n_passed;
    }
    
    { // test 2
        double I = numerics::simpson_integral<double>(test_func<double>, 0.0, 1.0, 1e-6);
        if (std::abs(I - val) > 1e-6) {
            std::cout << "simpson_integral failed double prec. test\n";
            ++n_failed;
        }
        else ++n_passed;
    }
    
    { // test 3
        float I = numerics::lobatto_integral<float>(test_func<float>, 0.0f, 1.0f, 1e-3f);
        if (std::abs(I - val) > 1e-3) {
            std::cout << "lobatto_integral failed single prec. test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    { // test 4
        double I = numerics::lobatto_integral<double>(test_func<double>, 0.0, 1.0, 1e-6);
        if (std::abs(I - val) > 1e-6) {
            std::cout << "lobatto_integral failed double prec. test\n";
            ++n_failed;
        }
        else ++n_passed;
    }

    std::cout << n_passed << "/" << n_passed + n_failed << " tests passed.\n";

    return 0;
}