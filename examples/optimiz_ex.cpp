#include "../numerics.hpp"

// g++ -Wall -g -o optimiz_ex examples/optimiz_ex.cpp optimiz.cpp newton.cpp roots.cpp grad.cpp eps.cpp -larmadillo

using namespace numerics;

int main() {
    arma::arma_rng::set_seed_random();
    arma::vec x(50, arma::fill::ones);
    x = -2.9*x;

    arma::vec y = x;
    arma::vec z = x;
    arma::vec w = x;

    //Styblinski-Tang function, guesses should be -5 < x_i < 5, all x_i = -2.90, for any size vector
    auto g = [](const arma::vec& x){ return arma::accu(arma::pow(x,4)-16*arma::pow(x,2)+5*x)/2; };
    auto dg = [](const arma::vec& x){ arma::vec y = 2*arma::pow(x,3) - 16*x + 2.5; return y; };
    
    std::cout << "initial value: " << g(x) << std::endl << std::endl;

    double f;
    clock_t t = clock();
    f = sgd(g,dg,z);
    t = clock() - t;
    std::cout << "\nsgd() results:\t\t\t\t\t" << f << std::endl << "sgd() took " << (float)t/CLOCKS_PER_SEC << " seconds" << std::endl;

    t = clock();
    f = sgd(g,w);
    t = clock() - t;
    std::cout << "\nsgd() without gradient results:\t\t\t" << f << std::endl << "sgd() took " << (float)t/CLOCKS_PER_SEC << " seconds" << std::endl;
    return 0;
}