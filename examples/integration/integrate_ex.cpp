#include "numerics.hpp"

// g++ -Wall -g -o integrate integrate_ex.cpp -lnumerics -larmadillo

using namespace numerics;

int main() {
    std::cout << "we will now try to integerate exp[-x^2] over [0,1]" << std::endl << std::endl;
    
    auto f = [](double x) -> double {return std::exp(-std::pow(x,2));}; // f(x) = exp[-x^2]
    
    long double val = 0.746824132812427025;
    double I = integrate(f,0,1);
    std::cout << "|error| < machine epsilon" << std::endl;
    std::cout << "\tintegrate() estimate: " << I << std::endl << "\tactual value: " << val << std::endl
              << "\terror: " << std::abs(I - val) << std::endl;
    
    I = integrate(f, 0, 1, LOBATTO, 1);
    std::cout << "|error| < 1" << std::endl;
    std::cout << "\tintegrate() estimate: " << I << std::endl << "\tactual value: " << val << std::endl
              << "\terror: " << std::abs(I - val) << std::endl; 

    std::cout << "\nnow we compare Simpson, Lobatto, and Chebyshev for the function: tan[sin x] + 2 over [0,2*pi]" << std::endl;

    auto g = [](double x){ return std::tan(std::sin(x))+2; }; // g(x) = tan[sin x] + 2
    val = 12.566370614359;
    std::cout << "using |err| < 0.001" << std::endl;
    double y;
    clock_t t = clock();
    y = simpson_integral(g, 0, 2*M_PI, 1e-3);
    t = clock() - t;
    std::cout << "simpson_integral() approx: " << y << " it took " << (float)t/CLOCKS_PER_SEC << " secs" << std::endl;
    std::cout << "\ttrue error: " << std::abs(val - y) << std::endl;

    t = clock();
    y = lobatto_integral(g, 0, 2*M_PI, 1e-3);
    t = clock() - t;
    std::cout << "lobatto_integral() approx: " << y << " it took " << (float)t/CLOCKS_PER_SEC << " secs" << std::endl;
    std::cout << "\ttrue error: " << std::abs(val - y) << std::endl;

    t = clock();
    y = chebyshev_integral(g, 0, 2*M_PI);
    t = clock() - t;
    std::cout << "chebyshev_integral() approx: " << y << " it took " << (float)t/CLOCKS_PER_SEC << " secs" << std::endl;
    std::cout << "\ttrue error: " << std::abs(val - y) << std::endl;

    std::cout << "\nsimpson_integral() stores intermediate function evaluations so the function is never evaluated twice at the same point, but may take longer to converge." << std::endl;
    std::cout << "\nlobatto_integral() is very accurate for smooth functions, but not recommended for discontinuous functions." << std::endl;
    std::cout << "\nchebyshev_integral() is spectrally accurate for smooth (analytic) functions, but not recommended for discontinuous functions." << std::endl;

    return 0;
}