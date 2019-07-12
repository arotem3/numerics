#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o knn_classifier knn_classifier_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

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

    numerics::knn_algorithm alg;
    alg = numerics::KD_TREE; // stores data in kd-tree allowing for O(log n) query time when x.n_cols << x.n_rows
    // alg = numerics::BRUTE; // stores data as is ==> O(n) query time
    numerics::knn_metric metr;
    metr = numerics::CONSTANT; // takes the average of all the neighbors
    // metr = numerics::DISTANCE; // weighs the average accoriding to distance
    arma::uvec k_set = {1,2,4,8,16,32};
    numerics::knn_classifier model(k_set,alg,metr);
    model.fit(X,Y);

    arma::vec xgrid = arma::linspace(-M_PI,M_PI,1000);
    arma::mat p = model.predict_probabilities(xgrid);
    arma::umat categories = model.predict_categories(X);

    arma::umat yhat = model.predict_categories(X);
    double prec = arma::sum(yhat.col(0) == Y.col(0) && yhat.col(0)) / (double)arma::sum(yhat.col(0));

    std::cout << "optimal k: " << model.num_neighbors() << '\n'
              << "Precision : " << prec << '\n'
              << "cross validation table:\n\tk\tF1\n" << model.get_cv_results();

    auto xx = conv(X);
    auto y1 = conv(Y.col(0));
    auto xxgrid = conv(xgrid);
    auto p1 = conv(p.col(0));
    auto c1 = conv(categories.col(0));

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