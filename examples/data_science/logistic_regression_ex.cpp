#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o logistic_regression logistic_regression_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

std::vector<double> conv(const arma::mat& u) {
    return arma::conv_to<std::vector<double>>::from(u);
}
std::vector<double> conv(const arma::umat& u) {
    return arma::conv_to<std::vector<double>>::from(u);
}

int main() {
    arma::mat X = (2*arma::randu(100,1) - 1)*M_PI;
    arma::umat Y_bool = (arma::abs(X) > 1.5);
    
    arma::mat Y = arma::zeros(Y_bool.n_rows,2);
    for (uint i=0; i < Y.n_rows; ++i) {
        if (Y_bool(i)) {
            Y(i,0) = 1;
            Y(i,1) = 0;
        } else {
            Y(i,0) = 0;
            Y(i,1) = 1;
        }
    }

    numerics::logistic_regression model;
    model.fit(X,Y);

    arma::vec xgrid = arma::linspace(-M_PI,M_PI,1000);
    arma::mat p = model.predict_probabilities(xgrid);
    arma::umat categories = model.predict_categories(X);

    std::cout << model.get_cv_results();

    auto xx = conv(X);
    auto y1 = conv(Y.col(0));
    auto xxgrid = conv(xgrid);
    auto p1 = conv(p.col(0));
    auto c1 = conv(categories.col(0));

    matplotlibcpp::named_plot("observed categories",xx,y1,"o");
    matplotlibcpp::named_plot("fit",xxgrid,p1,"-");
    matplotlibcpp::named_plot("predicted categories",xx,c1,".");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}