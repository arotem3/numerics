#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o kde kde_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

using namespace numerics;
std::vector<double> to_vec(const arma::vec& x) {
    return arma::conv_to<std::vector<double>>::from(x);
}

const int sample_size = 300;

int main() {
    arma::arma_rng::set_seed(123);

    arma::vec sample(sample_size);
    sample.rows(0,sample_size/2-1) = arma::randn(sample_size/2);
    sample.rows(sample_size/2,sample_size-1) = 2*arma::randn(sample_size/2) + 10;

    kernels K = kernels::gaussian; // gaussian, square, triangle, parabolic
    bandwidth_estimator estimator = bandwidth_estimator::rule_of_thumb_sd; // rule_of_thumb_sd, min_sd_iqr, direct_plug_in, grid_cv
    bool enable_binning = true;
    kde pdf_hat(K, estimator, enable_binning);
    // kde pdf_hat(0.8, K, enable_binning);
    pdf_hat.fit(sample);

    arma::vec x = arma::linspace(-5,20,1000);
    arma::vec p_hat = pdf_hat.predict(x);

    std::cout << "selected bandwidth: " << pdf_hat.bandwidth << std::endl;

    int N = 1000;
    arma::vec sample_new = pdf_hat.sample(N); // we can resample using our KDE

    int n_bins = 30;

    auto xx = to_vec(x);
    auto hist = to_vec(sample);
    auto bootstrap = to_vec(sample_new);
    auto p_hat0 = to_vec(sample_size*p_hat);

    matplotlibcpp::named_hist("sample", hist, n_bins);
    matplotlibcpp::named_plot("kde",xx, p_hat0, "-k");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    p_hat0 = to_vec(N*p_hat);
    matplotlibcpp::named_hist("bootstrap sample", bootstrap, n_bins);
    matplotlibcpp::named_plot("kde",xx, p_hat0, "-k");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}