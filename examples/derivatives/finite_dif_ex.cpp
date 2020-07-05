#include "numerics.hpp"

// g++ -Wall -g -o finite_dif finite_dif_ex.cpp -O3 -lnumerics -larmadillo

using namespace numerics;

int main() {
    auto f = [](double x) -> double {return std::sin(x);};
    auto df = [](double x) -> double {return std::cos(x);};

    double df0 = df(0.5);
    double df1 = deriv(f,0.5);

    std::cout << "the derivative of sin(0.5) is: " << df0 << std::endl
              << "\tthe approximation of that derivative with h = 1e-2 is: " << df1 << std::endl
              << "\ttrue error: " << std::abs(df0 - df1) << std::endl;
    
    auto g = [](const arma::vec& x) -> double {return arma::norm(x);};
    auto dg = [](const arma::vec& x) -> arma::vec {
        arma::vec v = arma::zeros<arma::vec>(3);
        v(0) = x(0)/arma::norm(x);
        v(1) = x(1)/arma::norm(x);
        v(2) = x(2)/arma::norm(x);
        return v;
    };
    arma::vec x = {0.4, 0.5, 0.33};

    arma::mat dg1 = grad(g, x).t();
    arma::mat dg0 = dg(x).t();

    std::cout << "g(x,y,z) = sqrt(x^2 + y^2 + z^2)" << std::endl
              << "\tthe true gradient of g([0.4   0.5   0.33]) is:" << dg0
              << "\tthe approximate gradient of g with h = 1e-2 is:" << dg1
              << "\t\t||true error||: " << arma::norm(dg0 - dg1, "Inf") << std::endl;
    
    return 0;
    
}