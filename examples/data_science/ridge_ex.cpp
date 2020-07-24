#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o ridge ridge_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

using namespace numerics;
namespace plt = matplotlibcpp;

typedef std::vector<double> dvec;

arma::mat f(arma::vec& X) {
    arma::vec y = arma::zeros(arma::size(X));
    for (int i=1; i < 10; ++i) {
        y += arma::sin(i*X)/i;
    }
    return y/M_PI;
}

int main() {
    // arma::arma_rng::set_seed(123);
    int n_obs = 100;
    arma::vec X = 5*arma::randu(n_obs) - 2.5;
    arma::vec y = f(X) + 0.15*arma::randn(n_obs);

    uint n_poly = 50;
    arma::vec centers = arma::linspace(-2.5,2.5,n_poly);
    arma::mat features = numerics::cubic_kernel(centers, X);

    arma::mat c_overfit = arma::solve(features, y);
    arma::vec xgrid = arma::linspace(-2.5, 2.5, 500);
    arma::mat features_grid = numerics::cubic_kernel(centers, xgrid);
    arma::vec yhat_overfit = features_grid * c_overfit;

    numerics::RidgeCV model;
    model.fit(features, y);
    arma::vec yhat = model.predict(features_grid);

    std::cout << "lambda: " << model.lambda << std::endl
              << "R^2: " << model.score(features,y) << std::endl
              << "effective degrees of freedom: " << model.eff_df << std::endl;

    dvec xx = arma::conv_to<dvec>::from(X);
    dvec yy = arma::conv_to<dvec>::from(y);
    dvec xxg = arma::conv_to<dvec>::from(xgrid);
    dvec yoverfit = arma::conv_to<dvec>::from(yhat_overfit);
    dvec yyhat = arma::conv_to<dvec>::from(yhat);

    plt::plot(xx,yy,"o");
    plt::named_plot("overfit", xxg, yoverfit);
    plt::named_plot("regularized fit", xxg, yyhat);
    plt::ylim(-1.2, 1.2);
    plt::legend();
    plt::show();

    return 0;
}