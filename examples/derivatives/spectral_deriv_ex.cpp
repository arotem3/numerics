#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o spectralD spectral_deriv_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

typedef std::vector<double> ddvec;

double f(double x) {
    return std::exp(-x*x);
}

arma::vec df(arma::vec x) {
    return (-2*x) % arma::exp(-x%x);
}

int main() {
    double a = -3; double b = 3; double m = 50;

    arma::vec x = arma::linspace(a,b);
    arma::vec y = df(x);

    numerics::PolyInterp dy = numerics::spectral_deriv(f,a,b,m);
    arma::vec v = dy.predict(x);

    std::cout << "max error : " << arma::norm(v - y, "inf") << std::endl;

    ddvec xx = arma::conv_to<ddvec>::from(x);
    ddvec yy = arma::conv_to<ddvec>::from(y);
    ddvec vv = arma::conv_to<ddvec>::from(v);

    matplotlibcpp::title("spectral derivative");
    matplotlibcpp::named_plot("actual derivative", xx, yy, "--k");
    matplotlibcpp::named_plot("spectral approximation", xx, vv, "-r");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}