#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o k_fold k_fold_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

typedef std::vector<double> dvec;

int main() {
    arma::vec x = 4*arma::randu(100)-2;
    arma::vec y = 2*x + arma::randn(100);

    int num_folds = 4;
    
    numerics::KFolds2Arr<double,double> split(num_folds);
    split.fit(x,y);

    std::map<std::string,std::string> keys_train;
        keys_train["marker"] = "o";
        keys_train["mfc"] = "none";
        keys_train["markersize"] = "4";
        keys_train["ls"] = "none";
        keys_train["label"] = "train data";
    std::map<std::string,std::string> keys_test;
        keys_test["marker"] = "o";
        keys_test["markersize"] = "7";
        keys_test["label"] = "test data";
        keys_test["ls"] = "none";

    for (int i=0; i < num_folds; ++i) {
        dvec xtrain = arma::conv_to<dvec>::from( split.trainX(i) );
        dvec ytrain = arma::conv_to<dvec>::from( split.trainY(i) );
        dvec xtest  = arma::conv_to<dvec>::from( split.testX(i) );
        dvec ytest  = arma::conv_to<dvec>::from( split.testY(i) );
        matplotlibcpp::subplot(2,2,i+1);
        matplotlibcpp::plot(xtrain, ytrain, keys_train);
        matplotlibcpp::plot(xtest,  ytest,  keys_test);
        matplotlibcpp::title("fold #"+std::to_string(i+1));
        if (i==0) matplotlibcpp::legend();
    }
    matplotlibcpp::tight_layout();
    matplotlibcpp::show();

    return 0;
}