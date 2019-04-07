#include "numerics.hpp"

// g++ -Wall -g -o integrate examples/integrate_ex.cpp -lnumerics -larmadillo

using namespace numerics;

int main() {
    std::cout << "we will now try to integerate exp[-x^2] over [0,1]" << std::endl << std::endl;
    
    auto f = [](double x) -> double {return std::exp(-std::pow(x,2));}; // f(x) = exp[-x^2]
    
    long double val = 0.746824132812427025;
    double I = integrate(f,0,1);
    std::cout << "|error| < machine epsilon" << std::endl;
    std::cout << "\tintegrate() estimate: " << I << std::endl << "\tactual value: " << val << std::endl
              << "\terror: " << std::abs(I - val) << std::endl;
    
    I = simpson_integral(f,0,1,1);
    std::cout << "|error| < 1" << std::endl;
    std::cout << "\tintegrate() estimate: " << I << std::endl << "\tactual value: " << val << std::endl
              << "\terror: " << std::abs(I - val) << std::endl; 

    std::cout << "\nnow we compare integrate(), tIntegrate(), and gIntegrate() for the periodic function: tan[sin x] + 2 over [0,1]" << std::endl;

    auto g = [](double x){ return std::tan(std::sin(x))+2; }; // g(x) = tan[sin x] + 2
    val = 12.566370614359;
    std::cout << "using |err| < 0.001" << std::endl;
    double y;
    clock_t t = clock();
    y = simpson_integral(g, 0, 2*M_PI, 0.001);
    t = clock() - t;
    std::cout << "simpson_integral() approx: " << y << " it took " << (float)t/CLOCKS_PER_SEC << " secs" << std::endl;
    std::cout << "\ttrue error: " << std::abs(val - y) << std::endl;

    t = clock();
    y = lobatto_integral(g,0,2*M_PI, 0.001);
    t = clock() - t;
    std::cout << "lobatto_integral() approx: " << y << " it took " << (float)t/CLOCKS_PER_SEC << " secs" << std::endl;
    std::cout << "\t true error: " << std::abs(val - y) << std::endl;

    std::cout << "\nsimpson_integral() is more efficient in how it evaluates the objective function." << std::endl;
    std::cout << "\nlobatto_integral() is very useful for easy to evaluate (and smooth) functions, but not recommended for 'bad' functions" << std::endl;

    return 0;
}