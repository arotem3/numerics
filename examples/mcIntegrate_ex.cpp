#include "numerics.hpp"

// g++ -g -Wall -o mcIntegrate examples/mcIntegrate_ex.cpp -lnumerics -larmadillo

using namespace numerics;

int main() {
    auto f = [](const arma::vec& u) -> double {
        double x = u(0);
        double y = u(1);
        return 2 + x - y + 4*x*y - 3*x*x -2*y*y;
    };

    arma::vec a = {0,0};
    arma::vec b = {1,1};

    std::cout << std::setprecision(5) << std::fixed;
    std::cout << "We will atempt to approximate the double integral of f(x,y) = 2 + x - y + 4xy - 3x^2 - 2y^2 over x{0,1} y{0,1}" << std::endl;
    std::cout << "The exact solution is: 4/3 i.e. " << 4.0/3 << std::endl;

    clock_t t = clock();
    double I = mcIntegrate(f,a,b);
    t = clock() - t;
    double err = std::abs(I - 4.0/3);
    std::string success = (err < 1e-2) ? ("yes") : ("no");

    std::cout << "with N = 1000 and ideal error bound err < 1e-2 our approximation yields:" << std::endl
              << "\tintegral = " << I << std::endl
              << "\terror = " << err << std::endl
              << "\tsuccess? " << success << std::endl
              << "\tcomputation time = " << (float)t/CLOCKS_PER_SEC << std::endl;

    t = clock();
    I = mcIntegrate(f,a,b,1e-2,10000);
    t = clock() - t;
    err = std::abs(I - 4.0/3);
    success = (err < 1e-2) ? ("yes") : ("no");

    std::cout << "with N = 10000 and ideal error bound err < 1e-2 our approximation yields:" << std::endl
              << "\tintegral = " << I << std::endl
              << "\terror = " << err << std::endl
              << "\tsuccess? " << success << std::endl
              << "\tcomputation time = " << (float)t/CLOCKS_PER_SEC << std::endl;

    t = clock();
    I = mcIntegrate(f,a,b,1e-3,1e4);
    t = clock() - t;
    err = std::abs(I - 4.0/3);
    success = (err < 1e-3) ? ("yes") : ("no");

    std::cout << "with N = 10000 and ideal error bound err < 1e-3 our approximation yields:" << std::endl
              << "\tintegral = " << I << std::endl
              << "\terror = " << err << std::endl
              << "\tsuccess? " << success << std::endl
              << "\tcomputation time = " << (float)t/CLOCKS_PER_SEC << std::endl;

    auto g = [](const arma::vec& u) -> double {
        double x = u(0);
        double y = u(1);
        double z = u(2);
        double r2 = x*x + y*y + z*z;

        if (r2 <= 1) return 1;
        else return 0;
    };
    double integral = 4*M_PI/3;

    std::cout << "We will now attempt to find the volume of a sphere of radius 1." << std::endl
              << "so we define f(x,y,z) as a piecewise function that returns 1 for all x^2 + y^2 + z^2 <= 1 and 0 everywhere else." << std::endl
              << "For monte carlo integration that places our domain on at least x{-1,1} y{-1,1} z{-1,1}." << std::endl
              << "the true solution is: 4pi/3 or approximately: " << integral << std::endl;
    
    a = {-1, -1, -1};
    b = {1, 1, 1};
    t = clock();
    I = mcIntegrate(g,a,b,1e-3,1e4);
    t = clock() - t;
    err = std::abs(I - integral);
    success = (err < 1e-2) ? ("yes") : ("no");

    std::cout << "\tintegral = " << I << std::endl
              << "\terror = " << err << std::endl
              << "\tsuccess? " << success << std::endl
              << "\tcomputation time = " << (float)t/CLOCKS_PER_SEC << std::endl;

    return 0;
}