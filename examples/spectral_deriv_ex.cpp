#include "numerics.hpp"
#include "plot.hpp"

// g++ -g -Wall -o spectralD examples/spectral_deriv_ex.cpp examples/wait.cpp -lnumerics -larmadillo

void wait_for_key(std::string s);

using namespace numerics;

double f(double x) {
    return std::exp(-x*x);
}

arma::vec df(arma::vec x) {
    return -2*x % arma::exp(-x%x);
}

int main() {
    double a = -3; double b = 3; double m = 200;

    arma::vec x = arma::linspace(a,b);
    arma::vec y = df(x);

    arma::vec u = {a,b};
    arma::vec v = specral_deriv(f,u,m);

    Gnuplot fig;
    fig.set_title("spectral derivative");
    lines(fig, x, y, "actual x,y");
    scatter(fig, u, v, "spectral approx",'r');

    std::cout << "max error : " << arma::norm(v - df(u), "inf") << std::endl;

    wait_for_key("Press ENTER to close...");

    return 0;
}