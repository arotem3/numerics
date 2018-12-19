#include "../numerics.hpp"
#include "gnuplot_i.hpp"


void wait_for_key(std::string s);

using namespace numerics;

typedef std::vector<double> stdv;

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

    stdv x0 = arma::conv_to<stdv>::from(x);
    stdv y0 = arma::conv_to<stdv>::from(y);
    stdv u0 = arma::conv_to<stdv>::from(u);
    stdv v0 = arma::conv_to<stdv>::from(v);

    Gnuplot fig("spectral derivative");
    fig.set_style("lines");
    fig.plot_xy(x0,y0,"actual x,y");
    fig.set_style("points");
    fig.plot_xy(u0,v0,"spectral approx");

    std::cout << "max error : " << arma::norm(v - df(u), "inf") << std::endl;

    wait_for_key("Press ENTER to close...");

    return 0;
}