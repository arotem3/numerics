#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o kNN knn_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

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
    arma::arma_rng::set_seed(123);
    arma::vec x = 5*arma::randu(100)-2.5;
    arma::vec y = f(x) + 0.1*arma::randn(arma::size(x))%arma::round(arma::randu(arma::size(x))*1);

    knn_algorithm alg = knn_algorithm::KD_TREE; // stores data in kd-tree allowing for O(log n) query time when x.n_cols << x.n_rows
                                                // BRUTE stores data as is ==> O(n) query time
    
    knn_metric metr = knn_metric::L2_DISTANCE; // L1_DISTANCE, CONSTANT
    arma::uvec k_set = arma::regspace<arma::uvec>(2,20);
    
    numerics::knn_regression model(k_set, alg, metr);
    model.fit(x,y);
    
    arma::vec t = arma::linspace(-3,3,500);
    arma::vec yhat = model.predict(t);

    std::cout << "optimal k: " << model.num_neighbors << '\n'
              << "RMSE : " << arma::norm(f(t) - yhat)/t.n_elem << '\n';

    ddvec xx = arma::conv_to<ddvec>::from(x);
    ddvec yy = arma::conv_to<ddvec>::from(y);
    ddvec tt = arma::conv_to<ddvec>::from(t);
    ddvec uu = arma::conv_to<ddvec>::from(yhat);
    ddvec vv = arma::conv_to<ddvec>::from(f(t));

    matplotlibcpp::plot(xx,yy,"or");
    matplotlibcpp::named_plot("kNN fit", tt, uu, "-b");
    matplotlibcpp::named_plot("actual function", tt, vv, "--m");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}