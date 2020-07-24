#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o knn_classifier knn_classifier_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

std::vector<double> conv(const arma::mat& u) {
    return arma::conv_to<std::vector<double>>::from(u);
}
std::vector<double> conv(const arma::umat& u) {
    return arma::conv_to<std::vector<double>>::from(u);
}

int main() {
    int sample_size = 100;
    arma::mat X = arma::zeros(sample_size,1);
    X.rows(0,sample_size/2-1) = arma::randn(sample_size/2)-1.5;
    X.rows(sample_size/2,sample_size-1) = arma::randn(sample_size/2)+1.5;
    arma::uvec Y = arma::zeros<arma::uvec>(sample_size);
    Y.rows(sample_size/2,sample_size-1) += 1;

    bool distance_weights = false;
    arma::uvec k_set = arma::regspace<arma::uvec>(2,20);
    numerics::KNeighborsClassifier model(k_set, 2, distance_weights);
    model.fit(X,Y);

    arma::vec xgrid = arma::linspace(-M_PI,M_PI,1000);
    arma::mat p = model.predict_proba(xgrid);
    arma::uvec pred_categories = model.predict(X);

    std::cout << "optimal k: " << model.k << '\n'
              << "accuracy : " << model.score(X, Y) << '\n';

    auto xx = conv(X);
    auto y1 = conv(Y);
    auto xxgrid = conv(xgrid);
    auto p1 = conv(p.col(1));
    auto c1 = conv(pred_categories);

    std::map<std::string,std::string> keys;
        keys["marker"] = "o";
        keys["ls"] = "none";
        keys["label"] = "actual categories";
    matplotlibcpp::plot(xx,y1,keys);
        keys["marker"] = "none";
        keys["ls"] = "-";
        keys["label"] = "predicted probabilities";
    matplotlibcpp::plot(xxgrid,p1,"-");
        keys["marker"] = "o";
        keys["mfc"] = "none";
        keys["ls"] = "none";
        keys["markersize"] = "10";
        keys["label"] = "predicted categories";
    matplotlibcpp::plot(xx,c1,keys);
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}