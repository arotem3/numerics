#include "../numerics.hpp"
#include "gnuplot_i.hpp"

// g++ -Wall -g -o kmeans_ex examples/kmeans_ex.cpp examples/wait.cpp kmeans.cpp -larmadillo

using namespace numerics;

void wait_for_key();

int main() {
    arma::arma_rng::set_seed_random();
    arma::mat a1 = 3 + arma::randn(2,100);
    arma::mat a2 = arma::randn(2,100);
    arma::mat A = arma::join_rows(a1,a2);
    kmeans kmu(A,2);

    arma::rowvec x = kmu.getClusters();
    // arma::mat c0 = A.cols( arma::find(x == 0) ); // find all data in A that is in cluster 0
    // arma::mat c1 = A.cols( arma::find(x == 1) ); // find all data in A that is in cluster 1
    arma::mat c0 = kmu[0]; // same as A.cols( arma::find(x == 0) ) also same as kmu.all_from_cluster(0);
    arma::mat c1 = kmu[1];

    Gnuplot fig1("test");
    
    typedef std::vector<double> stdv;
    stdv u0 = arma::conv_to<stdv>::from(c0.row(0));
    stdv v0 = arma::conv_to<stdv>::from(c0.row(1));
    stdv u1 = arma::conv_to<stdv>::from(c1.row(0));
    stdv v1 = arma::conv_to<stdv>::from(c1.row(1));

    fig1.set_style("points").plot_xy(u0,v0);
    fig1.plot_xy(u1,v1);

    kmu.print(std::cout);

    wait_for_key();

    return 0;
}