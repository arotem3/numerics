#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -Wall -g -o kmeans kmeans_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

using namespace numerics;

typedef std::vector<double> ddvec;

int main() {
    arma::arma_rng::set_seed_random();
    arma::mat a1 = 3 + arma::randn(100,2);
    arma::mat a2 = arma::randn(100,2);
    arma::mat A = arma::join_cols(a1,a2);
    kmeans kmu(A,2);
    
    kmu.help();

    arma::mat c0 = kmu[0]; // same as above also same as kmu.all_from_cluster(0);
    arma::mat c1 = kmu[1];

    kmu.summary(std::cout);

    ddvec c0x = arma::conv_to<ddvec>::from(c0.col(0));
    ddvec c0y = arma::conv_to<ddvec>::from(c0.col(1));
    ddvec c1x = arma::conv_to<ddvec>::from(c1.col(0));
    ddvec c1y = arma::conv_to<ddvec>::from(c1.col(1));

    matplotlibcpp::named_plot("cluster 0", c0x, c0y,"o");
    matplotlibcpp::named_plot("cluster 1", c1x, c1y,"o");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}