#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o spectralD examples/spectral_deriv_ex.cpp -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

using namespace numerics;
typedef std::vector<double> ddvec;

double f(double x) {
    return std::exp(-x*x);
}

arma::vec df(arma::vec x) {
    return -2*x % arma::exp(-x%x);
}

int main() {
    double a = -3; double b = 3; double m = 30;

    arma::vec x = arma::linspace(a,b);
    arma::vec y = df(x);

    arma::vec u = {a,b};
    arma::vec v = specral_deriv(f,u,m);

    std::cout << "max error : " << arma::norm(v - df(u), "inf") << std::endl;

    ddvec xx = arma::conv_to<ddvec>::from(x);
    ddvec yy = arma::conv_to<ddvec>::from(y);
    ddvec uu = arma::conv_to<ddvec>::from(u);
    ddvec vv = arma::conv_to<ddvec>::from(v);

    matplotlibcpp::title("spectral derivative");
    matplotlibcpp::named_plot("actural derivative", xx, yy, "--k");
    matplotlibcpp::named_plot("spectral approximation", uu, vv, "-r");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}