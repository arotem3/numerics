#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o logistic_regression logistic_regression_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

std::vector<double> conv(const arma::mat& u) {
    return arma::conv_to<std::vector<double>>::from(u);
}
std::vector<double> conv(const arma::umat& u) {
    return arma::conv_to<std::vector<double>>::from(u);
}

int main() {
    arma::mat X = (2*arma::randu(100,1) - 1)*M_PI;
    arma::uvec y = (arma::abs(X)+0.1*arma::randn(arma::size(X)) > 1.5);

    numerics::PolyFeatures poly(3);
    arma::mat Xp3 = poly.fit_predict(X);
    numerics::LogisticRegression model;
    // model.set_lambda(1e-2);
    model.fit(Xp3,y);

    arma::vec xgrid = arma::linspace(-M_PI,M_PI,1000);
    arma::mat p = model.predict_proba(poly.predict(xgrid));
    arma::uvec yh = model.predict(Xp3);

    std::cout << "lambda:" << model.lambda << "\n"
              << "accuracy: " << model.score(Xp3,y) << "\n";

    auto xx = conv(X);
    auto y1 = conv(y);
    auto xxgrid = conv(xgrid);
    auto p1 = conv(p.col(1));
    auto c1 = conv(yh);

    matplotlibcpp::named_plot("observed classes",xx,y1,"o");
    matplotlibcpp::named_plot("fit",xxgrid,p1,"-");
    matplotlibcpp::named_plot("predicted classes",xx,c1,".");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}