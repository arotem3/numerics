#include <armadillo>
#include "numerics/optimization/simplex.hpp"

using numerics::optimization::simplex;

int main()
{
    arma::vec f = {1,3}; // z(x) = 1x + 3y
    arma::mat A = { {1,1}, // x + y
                    {5,2}, // 5x + 2y
                    {1,2}}; // x + 2y
    arma::vec b = { 10, // x + y <= 10
                    20, // 5x + 2y <= 20
                    36}; // x + 2y <= 36
    arma::vec x;
    double y = simplex(x, f, A, b);

    bool cons = arma::all(A*x <= b) and arma::all(x >= 0);

    if (not cons)
        std::cout << "simplex failed\n";
    else
        std::cout << "simplex succeeded\n";

    return 0;
}