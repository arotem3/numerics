#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o kernel_smooth kernel_smooth_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

using namespace numerics;
typedef std::vector<double> ddvec;

arma::vec f(arma::vec& X) {
    arma::vec y = arma::zeros(arma::size(X));
    for (int i=1; i < 10; ++i) {
        y += arma::sin(i*X)/i;
    }
    return 0.5 - y/M_PI;
}

int main() {
    arma::vec x = 5*arma::randu(200)-2.5;
    arma::vec y = f(x) + 0.1*arma::randn(arma::size(x));

    kernels k = kernels::gaussian; // square, triangle, parabolic
    bool data_to_bins = false;
    kernel_smooth model(k, data_to_bins);
    model.fit(x,y);
    arma::vec t = arma::linspace(-3,3,300);
    arma::vec yhat = model.predict(t);

    std::cout << "bandwidth : " << model.bandwidth << std::endl;
    double r2 = arma::norm(f(x) - model(x)) / arma::norm(f(x) - arma::mean(f(x)));
    r2 = 1 - r2*r2;
    std::cout << "'functional' r^2 : " << r2 << std::endl;

    ddvec xx = arma::conv_to<ddvec>::from(x);
    ddvec yy = arma::conv_to<ddvec>::from(y);
    ddvec tt = arma::conv_to<ddvec>::from(t);
    ddvec uu = arma::conv_to<ddvec>::from(yhat);
    ddvec vv = arma::conv_to<ddvec>::from(f(t));

    matplotlibcpp::plot(xx,yy,"or");
    matplotlibcpp::named_plot("kernel fit", tt, uu, "-b");
    matplotlibcpp::named_plot("actual function", tt, vv, "--m");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}