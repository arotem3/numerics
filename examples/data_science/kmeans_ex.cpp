#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -Wall -g -o kmeans kmeans_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

using namespace numerics;

typedef std::vector<double> ddvec;

int main() {
    arma::arma_rng::set_seed_random();
    arma::mat A = arma::randn(200,2);
    int k = 4;
    numerics::kmeans kmu(A,k);
    
    // kmu.help();

    kmu.summary(std::cout);

    for (int i=0; i < k; ++i) {
        arma::mat cluster_i = kmu[i];// same as above also same as kmu.all_from_cluster(i);
        ddvec x = arma::conv_to<ddvec>::from(cluster_i.col(0));
        ddvec y = arma::conv_to<ddvec>::from(cluster_i.col(1));
        matplotlibcpp::named_plot("cluster " + std::to_string(i), x, y, "o");
    }
    
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}