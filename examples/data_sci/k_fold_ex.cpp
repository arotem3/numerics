#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o k_fold k_fold_ex.cpp -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

using namespace numerics;
namespace plt = matplotlibcpp;
typedef std::vector<double> ddvec;

int main() {
    arma::vec x = 4*arma::randu(100)-2;
    arma::vec y = 2*x + arma::randn(100);

    int num_folds = 4;
    folds train_test = k_fold(x, y, num_folds);

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
        ddvec xx = arma::conv_to<ddvec>::from( train_test.at(i).X);
        ddvec yy = arma::conv_to<ddvec>::from( train_test.at(i).Y);
        ddvec xexclude = arma::conv_to<ddvec>::from( x.rows(train_test.at(i).exclude_indices));
        ddvec yexclude = arma::conv_to<ddvec>::from( y.rows(train_test.at(i).exclude_indices));
        plt::subplot(2,2,i+1);
        plt::plot(xexclude, yexclude, keys_B);
        plt::plot(xx, yy, keys_A);
        plt::title("fold #"+std::to_string(i+1));
        if (i==0) plt::legend();
    }
    plt::tight_layout();
    plt::show();

    return 0;
}