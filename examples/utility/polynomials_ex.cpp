#include "numerics.hpp"

// g++ -g -Wall -o polynomials polynomials_ex.cpp -O3 -lnumerics -larmadillo

int main() {
    arma::vec p = {1, 1, 1, 1}; // x^3 + x^2 + x + 1
    
    arma::vec dp = numerics::polyder(p); // should return 3x^2 + 2x + 1
    arma::vec d2p = numerics::polyder(p,2); // should return 6x + 2

    arma::vec ip = numerics::polyint(p); // should return x^4/4 + x^3/3 + x^2/2 + x + 0
    arma::vec ip_2 = numerics::polyint(p,2); // should return x^4/4 + x^3/3 + x^2/2 + x + 2

    std::cout << "original polynomial: " << p.t() << std::endl
              << "first derivative: " << dp.t() << std::endl
              << "second derivative: " << d2p.t() << std::endl
              << "integral (with coefficient +0):" << ip.t() << std::endl
              << "integral (with coefficient +2):" << ip_2.t() << std::endl;

    return 0;
}