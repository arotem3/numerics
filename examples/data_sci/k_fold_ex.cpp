#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o k_fold k_fold_ex.cpp -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

using namespace numerics;
typedef std::vector<double> ddvec;

int main() {
    arma::vec x = 4*arma::randu(100)-2;
    arma::vec y = 2*x + arma::randn(100);

    int num_folds = 3;
    folds train_test = k_fold(x, y, num_folds);

    for (int i=0; i < num_folds; ++i) {
        ddvec xx = arma::conv_to<ddvec>::from( train_test.at(i).X);
        ddvec yy = arma::conv_to<ddvec>::from( train_test.at(i).Y);
        matplotlibcpp::named_plot("fold #"+std::to_string(i+1), xx, yy, "o");
    }
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}