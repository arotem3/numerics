#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o splines splines_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

using namespace numerics;
typedef std::vector<double> ddvec;

arma::vec f(const arma::vec& X) {
    arma::vec y = arma::zeros(arma::size(X));
    for (int i=1; i < 10; ++i) {
        y += arma::sin(i*X)/i;
    }
    return 0.5 - y/M_PI;
}

int main() {
    arma::arma_rng::set_seed_random();
    arma::mat X = 5*arma::randu(100,1) - 2.5;
    arma::vec y = f(X) + 0.05*arma::randn(100,1);

    numerics::Splines model;
    // model.set_df(15);
    // model.set_lambda(c1);
    model.fit(X,y);
    std::cout << "lambda : " << model.lambda << std::endl
              << "df : " << model.eff_df << std::endl;

    int N = 200;
    arma::mat xgrid = arma::linspace(-2.5,2.5,N);
    arma::vec yHat = model.predict(xgrid);

    ddvec X1 = arma::conv_to<ddvec>::from(X);
    ddvec Y1 = arma::conv_to<ddvec>::from(y);
    ddvec xx = arma::conv_to<ddvec>::from(xgrid);
    ddvec yy = arma::conv_to<ddvec>::from(yHat);
    ddvec ff = arma::conv_to<ddvec>::from(f(xgrid));

    matplotlibcpp::plot(X1, Y1, "ok");
    matplotlibcpp::named_plot("spline fit",xx,yy, "-b");
    matplotlibcpp::named_plot("actual function", xx, ff,"--m");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}