#include "numerics.hpp"

// g++ -g -Wall -o simplex simplex_ex.cpp -O3 -lnumerics -larmadillo

int main() {
    arma::vec f = {1,3}; // z(x) = 1x + 3y
    arma::mat conRHS = {{1,1}, // x + y
                        {5,2}, // 5x + 2y
                        {1,2}}; // x + 2y
    arma::vec conLHS = {10, // x + y <= 10
                        20, // 5x + 2y <= 20
                        36}; // x + 2y <= 36
    arma::vec x;
    double y = numerics::optimization::simplex(x, f, conRHS, conLHS);
    std::cout << "max val: " << y << std::endl << "occuring at:\n" << x.t() << std::endl;
    return 0;
}