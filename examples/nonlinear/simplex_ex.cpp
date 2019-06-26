#include "numerics.hpp"

// g++ -g -Wall -o simplex simplex_ex.cpp -O3 -lnumerics -larmadillo

using namespace numerics;

int main() {
    arma::mat A = {{1,1,1,0,0,0,10},
                   {5,2,0,1,0,0,20},
                   {1,2,0,0,1,0,36},
                   {-1,-3,0,0,0,1,0}};
    std::cout << A << std::endl;
    arma::vec x;
    double y = simplex(A,x);
    std::cout << "max val: " << y << std::endl << "occuring at:\n" << x << std::endl;

    arma::rowvec f = {1,3}; // z(x) = 1x + 3y
    arma::mat conRHS = {{1,1}, // x + y
                        {5,2}, // 5x + 2y
                        {1,2}}; // x + 2y
    arma::vec conLHS = {10, // x + y <= 10
                        20, // 5x + 2y <= 20
                        36}; // x + 2y <= 36
    arma::vec x1;
    double y1 = simplex(f,conRHS,conLHS,x1);
    std::cout << "max val: " << y1 << std::endl << "occuring at:\n" << x1 << std::endl;
    return 0;
}