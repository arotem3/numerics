#include "numerics.hpp"

// g++ -Wall -g -o genetic examples/genetic_ex.cpp -lnumerics -larmadillo

using namespace numerics;

int main() {
    auto f = [](const arma::vec& x){ return -arma::accu(arma::pow(x,4)-16*arma::pow(x,2)+5*x)/2; };
    arma::vec xMin = {-5,-5,-5,-5};
    arma::vec xMax = {5,5,5,5};
    arma::vec x;
    std::cout << "Let's try to optimize f(x) = sum(-x^4 - 16x^2 + 5x)/2 where x is a 4 dimensional vector." << std::endl
              << "We will find the global max of the constrained problem [-5 -5 -5 -5] <= x <= [5  5  5  5]." << std::endl;

    clock_t t = clock();
    double z = genOptim(f,x,xMin,xMax);
    t = clock() - t;
    std::cout << "global maximum at: " << x.t() << "the value is: " << z << std::endl;
    std::cout << "computation time: " << (float)t/CLOCKS_PER_SEC << std::endl << std::endl;

    std::cout << "We will try to solve the unconstrained problem with initial guess x = [-2.5  -3  -3.1  -2.8] and search radius of 1.0" << std::endl;
    arma::vec x0 = {-2.5,-3,-3.1,-2.8};
    t = clock();
    z = genOptim(f,x0);
    t = clock() - t;
    std::cout << "local max at: " << x0.t() << "the value is: " << z << std::endl;
    std::cout << "computation time: " << (float)t/CLOCKS_PER_SEC << std::endl << std::endl;

    double u = -2.904;
    arma::vec U = {u,u,u,u};
    std::cout << "actual max at: " << U.t() << "actual max is: " << f(U) << std::endl;

    // boolean optimization
    std::cout << std::endl << "Let's try to solve the bolean optimization problem g(x) = sum(x) where x is a 64 dimensional boolean vector." << std::endl;
    auto g = [](const arma::uvec& x) -> double {return arma::accu(x);};
    arma::uvec v;
    t = clock();
    z = boolOptim(g,v,64);
    t = clock() - t;
    std::cout << "global max estimate value is: " << z << std::endl
              << "computation time: " << (float)t/CLOCKS_PER_SEC << std::endl
              << "true solution: [1 1 ... 1] with value 64." << std::endl;

    return 0;
}