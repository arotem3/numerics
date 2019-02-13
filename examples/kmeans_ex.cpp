#include "numerics.hpp"
#include "plot.hpp"

// g++ -Wall -g -o kmeans examples/kmeans_ex.cpp examples/wait.cpp -lnumerics -larmadillo

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
    arma::mat c0 = kmu[0]; // same as above also same as kmu.all_from_cluster(0);
    arma::mat c1 = kmu[1];

    Gnuplot fig;

    plot(fig, (arma::mat)c0.row(0), (arma::mat)c0.row(1), {{"legend","cluster 0"},{"linespec","o"}});
    plot(fig, (arma::mat)c1.row(0), (arma::mat)c1.row(1), {{"legend","cluster 1"},{"linespec","o"}});

    kmu.summary(std::cout);

    wait_for_key();

    return 0;
}