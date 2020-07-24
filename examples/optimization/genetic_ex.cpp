#include "numerics.hpp"

// g++ -Wall -g -o genetic genetic_ex.cpp -O3 -lnumerics -larmadillo

using namespace numerics;

int main() {
    auto f = [](const arma::vec& x){ return -arma::accu(arma::pow(x,4)-16*arma::pow(x,2)+5*x)/2; };
    arma::vec xMin = {-5,-5,-5,-5};
    arma::vec xMax = {5,5,5,5};
    arma::vec x;
    std::cout << "Let's try to maximize f(x) = sum(-x^4 - 16x^2 + 5x)/2 where x is a 4 dimensional vector." << std::endl
              << "We will find the global max of the constrained problem [-5 -5 -5 -5] <= x <= [5  5  5  5]." << std::endl;

    numerics::optimization::GeneticOptimizer genOptim;

    clock_t t = clock();
    genOptim.maximize(x, f, xMin, xMax);
    t = clock() - t;
    std::cout << "global maximum at: " << x.t() << "the value is: " << f(x) << std::endl;
    std::cout << "computation time: " << (float)t/CLOCKS_PER_SEC << std::endl << std::endl;

    std::cout << "We will try to solve the unconstrained problem with initial guess x = [0 0 0 0] and search radius of 2.0" << std::endl;
    arma::vec x0 = {0,0,0,0};
    genOptim.set_search_radius(2.0);
    t = clock();
    genOptim.maximize(x0, f);
    t = clock() - t;
    std::cout << "local max at: " << x0.t() << "the value is: " << f(x0) << std::endl;
    std::cout << "computation time: " << (float)t/CLOCKS_PER_SEC << std::endl << std::endl;

    double u = -2.904;
    arma::vec U = {u,u,u,u};
    std::cout << "actual max at: " << U.t() << "actual max is: " << f(U) << std::endl;

    return 0;
}