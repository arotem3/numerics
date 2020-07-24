#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o kNN knn_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

typedef std::vector<double> dvec;

arma::vec f(arma::vec& X) {
    arma::vec y = arma::zeros(arma::size(X));
    for (int i=1; i < 10; ++i) {
        y += arma::sin(i*X)/i;
    }
    return 0.5 - y/M_PI;
}

int main() {
    arma::arma_rng::set_seed(123);
    arma::vec x = 5*arma::randu(200)-2.5;
    arma::vec y = f(x) + 0.1*arma::randn(arma::size(x))%arma::round(arma::randu(arma::size(x))*1);

    bool distance_weights = false;
    arma::uvec k_set = arma::regspace<arma::uvec>(2,20);
    
    numerics::KNeighborsRegressor model(k_set,2,distance_weights);
    model.fit(x,y);
    
    arma::vec t = arma::linspace(-3,3,500);
    arma::vec yhat = model.predict(t);

    std::cout << "optimal k : " << model.k << '\n'
              << "model R^2 : " << model.score(x,y) << '\n';

    dvec xx = arma::conv_to<dvec>::from(x);
    dvec yy = arma::conv_to<dvec>::from(y);
    dvec tt = arma::conv_to<dvec>::from(t);
    dvec uu = arma::conv_to<dvec>::from(yhat);
    dvec vv = arma::conv_to<dvec>::from(f(t));

    matplotlibcpp::plot(xx,yy,"or");
    matplotlibcpp::named_plot("kNN fit", tt, uu, "-b");
    matplotlibcpp::named_plot("actual function", tt, vv, "--m");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}