#include <numerics.hpp>
#include <matplotlibcpp.h>

// g++ -g -Wall -O3 -o nn_regression nn_regression_ex.cpp -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

arma::mat f(arma::mat& X) {
    arma::mat y = arma::zeros(arma::size(X));
    for (int i=1; i < 5; ++i) {
        y += arma::sin(i*X)/i;
    }
    y = 0.5 - y/M_PI;
    return y;
}

using namespace numerics;
typedef std::vector<double> dvec;

int main() {
    int N = 1000;
    arma::mat x = 6*arma::randu(N,1) - 3;
    arma::vec y = f(x) + 0.2*arma::randn(arma::size(x));

    std::string loss = "mse";
    std::vector<std::pair<int,std::string>> layers = {{100,"relu"}};
    long max_iter = 200;
    double tol = 1e-4;
    double l2 = 1e-4;
    double l1 = 0;
    std::string optimizer = "adam";
    bool verbose = true;

    NeuralNetRegressor model(layers, loss, max_iter, tol, l2, l1, optimizer, verbose);
    model.fit(x,y);

    std::cout << "R^2 : " << std::fixed << std::setprecision(2) << model.score(x,y) << "\n";

    arma::vec t = arma::linspace(-3,3);
    arma::vec yh = model.predict(t);

    dvec xx = arma::conv_to<dvec>::from(x);
    dvec yy = arma::conv_to<dvec>::from(y);
    dvec tt = arma::conv_to<dvec>::from(arma::vec(t));
    dvec yyh = arma::conv_to<dvec>::from(arma::vec(yh.col(0)));

    std::map<std::string,std::string> ls = {{"label","data"},{"ls","none"},{"marker","o"}};
    matplotlibcpp::plot(xx, yy, ls);
    ls["label"] = "fit"; ls["marker"] = ""; ls["ls"] = "-";
    matplotlibcpp::plot(tt, yyh, ls);
    matplotlibcpp::show();

    return 0;
}