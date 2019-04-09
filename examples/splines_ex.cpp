#include "numerics.hpp"
#include "plot.hpp"

// g++ -g -Wall -o splines examples/splines_ex.cpp -lnumerics -larmadillo

using namespace numerics;

inline void wait() {
    std::cout << "Press ENTER to continue." << std::endl;
    std::cin.get();
}

int main() {
    arma::mat X = 4*arma::randu(100,2) - 2;
    arma::vec Y = arma::pow(X.col(0),2) + arma::pow(X.col(1),2) + 0.4*arma::randn(100,1);

    int m = 2;

    splines model(X, Y, m);
    std::cout << "lambda : " << model.smoothing_param() << std::endl
              << "gcv : " << model.gcv_score() << std::endl
              << "df : " << model.eff_df() << std::endl;

    int N = 30;
    arma::mat xgrid = arma::linspace(-4,4,N);
    xgrid = meshgrid(xgrid);
    xgrid = arma::join_rows(
        arma::vectorise(xgrid), arma::vectorise(xgrid.t())
    );
    arma::mat yHat = model(xgrid);

    Gnuplot fig;
    plot3d(fig, (arma::mat)xgrid.col(0), (arma::mat)xgrid.col(1), yHat);

    wait();

    return 0;
}