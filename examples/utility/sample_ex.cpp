#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -Wall -g -o sample_from sample_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

using namespace numerics;
typedef std::vector<double> ddvec;

const int n = 10;
const double p = 0.3;

int coef[n+1][n+1];

void choose() {
    for (int i=0; i <= n; ++i) {
        for (int j=0; j <= i; ++j) {
            if (j == 0 || j == i) coef[i][j] = 1;
            else coef[i][j] = coef[i - 1][j - 1] + coef[i - 1][j];
        }
    }
}

double binom_pdf(int k) {
    return coef[n][k] * std::pow(p, k) * std::pow(1-p, n-k);
}

int main() {
    choose();

    arma::vec x = arma::regspace(0,n);
    arma::vec pdf = arma::zeros(n+1);
    for (int i=0; i <= n; ++i) pdf(i) = binom_pdf(i);

    arma::vec sample = sample_from(1000, pdf);

    ddvec xx = arma::conv_to<ddvec>::from(arma::linspace(0,n));
    ddvec hist = arma::conv_to<ddvec>::from(sample);
    ddvec pdf0 = arma::conv_to<ddvec>::from(
        cubic_interp(x,1000*pdf)(arma::linspace(0,n))
        );

    matplotlibcpp::named_hist("sample", hist, 30);
    matplotlibcpp::named_plot("pdf",xx, pdf0, "-r");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}