#include "numerics.hpp"

// g++ -g -Wall -o fmin fmin_ex.cpp -lnumerics -larmadillo

double f(double x) {
    return 1 + std::exp(-100*x) + (0.99-x)/(x-1); // unimodal, very large derivative near minimum and singluar at one of the end points
}

int main() {
    double a=0, b=1; // bounds, appropriate to this function
    double x0=0;

    std::cout << "f(x) = 1 + exp(-100x) + (0.99-x)/(x-1)\n"
              << "\ttrue minimum: x = " << 0.0902125 << "\n"
              << "\tusing fminbnd: x = " << numerics::optimization::fminbnd(f,a,b) << "\n"
              << "\tusing fminsearch: x = " << numerics::optimization::fminsearch(f,x0) << "\n";
    return 0;
}