#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -Wall -g -o kmeans kmeans_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

typedef std::vector<double> ddvec;

arma::mat gen_balls(int n_balls, int n_obs) {
    arma::arma_rng::set_seed_random();
    arma::mat m = 5*arma::randn(n_balls,2);
    arma::mat data = arma::randn(n_obs,2);
    for (int i=0; i < n_balls; ++i) data.rows(i*n_obs/n_balls, (i+1)*n_obs/n_balls-1).each_row() += m.row(i);
    return data;
}

void plot_classes(int k, const arma::mat& data, const arma::uvec& labels, const arma::mat centroids) {
    arma::vec X = data.col(0), Y = data.col(1);
    for (int i=0; i < k; ++i) {
        arma::uvec cluster_i = arma::find(labels == i);
        ddvec x = arma::conv_to<ddvec>::from(X(cluster_i));
        ddvec y = arma::conv_to<ddvec>::from(Y(cluster_i));
        matplotlibcpp::named_plot("cluster " + std::to_string(i), x, y, "o");
        std::vector<double> cx = {centroids(i,0)}, cy = {centroids(i,1)};
        matplotlibcpp::plot(cx, cy, "*k");
    }
    matplotlibcpp::legend();
    matplotlibcpp::show();
}

int main() {
    int k = 4;
    arma::mat data = gen_balls(k, 1000);

    arma::uvec labels;
    arma::mat centroids;

    bool use_sgd = true;
    if (use_sgd) {
        numerics::kmeans_sgd kmu(k);
        int batch_size = 50, max_iter = 10;
        labels = kmu.fit(data, batch_size, max_iter);
        // centroids = kmu.clusters;
        centroids = kmu.points_nearest_centers; // alternative to using the means, this returns actual data points which may be useful when as a data reduction technique
    } else {
        numerics::kmeans kmu(k);
        double tol = 1e-5;
        labels = kmu.fit(data, tol);
        centroids = kmu.clusters;
    }

    plot_classes(k, data, labels, centroids);

    return 0;
}