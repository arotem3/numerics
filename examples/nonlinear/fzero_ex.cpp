#include "numerics.hpp"

// g++ -g -Wall -o fzero fzero_ex.cpp -lnumerics -larmadillo

using namespace numerics;
using std::exp;
using std::pow;

int main() {
    auto g = [](double x) -> double { return exp(x) - pow(x,2); };
    auto dg = [](double x) -> double { return exp(x) - 2*x; };
    double y;

    std::cout << "let's try to find roots of: f(x) = exp[x] - x^2" << std::endl;

    clock_t t = clock();
    y = newton(g, dg, 0);
    t = clock() - t;

    std::cout << "(1)\tfor newton's method we need the derivative of the function and a starting point:" << std::endl
              << "\t\tnewton() returned:     " << y << std::endl
              << "\t\t|f(root)|:             " << std::abs(g(y)) << std::endl
              << "\t\tnewton() took:         " << (float)t/CLOCKS_PER_SEC << " seconds" << std::endl << std::endl;
    
    t = clock();
    y = secant(g, -3, 3);
    t = clock() - t;

    std::cout << "(2)\tfor secant method we need to starting points:" << std::endl
              << "\t\tsecant() returned:     " << y << std::endl
              << "\t\t|f(root)|:             " << std::abs(g(y)) << std::endl
              << "\t\tsecant() took:         " << (float)t/CLOCKS_PER_SEC << " seconds" << std::endl << std::endl;

    t = clock();
    y = bisect(g, -3, 3);
    t = clock() - t;

    std::cout << "(3)\tfor bisection method we need to end points of a seach region:" << std::endl
              << "\t\tbisect() returned:     " << y << std::endl
              << "\t\t|f(root)|:             " << std::abs(g(y)) << std::endl
              << "\t\tsecant() took:         " << (float)t/CLOCKS_PER_SEC << " seconds" << std::endl << std::endl;
    
    t = clock();
    y = fzero(g, -3, 3);
    t = clock() - t;

    std::cout << "(4)\tfor roots() we need end points again but we adaptively use two methods to narrow our search region:" << std::endl
              << "\t\troots() returned:      " << y << std::endl
              << "\t\t|f(root)|:             " << std::abs(g(y)) << std::endl
              << "\t\troots() took:          " << (float)t/CLOCKS_PER_SEC << " seconds" << std::endl << std::endl;

    return 0;
}