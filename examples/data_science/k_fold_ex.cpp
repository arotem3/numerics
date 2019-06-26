#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o k_fold k_fold_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

using namespace numerics;
typedef std::vector<double> ddvec;

int main() {
    arma::vec x = 4*arma::randu(100)-2;
    arma::vec y = 2*x + arma::randn(100);

    int num_folds = 4;
    k_folds split(x, y, num_folds);

    std::map<std::string,std::string> keys_A;
        keys_A["marker"] = "o";
        keys_A["markersize"] = "7";
        keys_A["label"] = "extracted test data";
        keys_A["ls"] = "none";
    std::map<std::string,std::string> keys_B;
        keys_B["marker"] = "o";
        keys_B["mfc"] = "none";
        keys_B["markersize"] = "4";
        keys_B["ls"] = "none";

    for (int i=0; i < num_folds; ++i) {
        ddvec xx = arma::conv_to<ddvec>::from( split[i] ); // or split.fold_X(i)
        ddvec yy = arma::conv_to<ddvec>::from( split(i) ); // or split.fold_Y(i)
        ddvec xexclude = arma::conv_to<ddvec>::from( split[-1-i] ); // or split.not_fold_X(i)
        ddvec yexclude = arma::conv_to<ddvec>::from( split(-1-i) ); // or split.not_fold_Y(i)
        matplotlibcpp::subplot(2,2,i+1);
        matplotlibcpp::plot(xexclude, yexclude, keys_B);
        matplotlibcpp::plot(xx, yy, keys_A);
        matplotlibcpp::title("fold #"+std::to_string(i+1));
        if (i==0) matplotlibcpp::legend();
    }
    matplotlibcpp::tight_layout();
    matplotlibcpp::show();

    return 0;
}