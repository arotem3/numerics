#include "numerics.hpp"
#include "plot.hpp"

// g++ -Wall -g -o sample_from examples/sample_ex.cpp examples/wait.cpp -lnumerics -larmadillo

using namespace numerics;

void wait_for_key();

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
    arma::vec hist = arma::zeros(n+1);
    for (int i=0; i <= n; ++i) hist(i) = arma::sum(sample==i);
    hist /= arma::sum(hist);

    Gnuplot fig;
    plot(fig, x, hist, {{"linespec","-or"},{"legend","sample"}});
    plot(fig, x, pdf, {{"linespec","-sb"},{"legend","pdf"}});

    wait_for_key();

    return 0;
}