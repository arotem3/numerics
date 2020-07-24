#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o kde kde_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

typedef std::vector<double> dvec;

int main() {
    arma::arma_rng::set_seed(123);

    int sample_size = 300;

    arma::vec sample(sample_size);
    sample.rows(0,sample_size/2-1) = arma::randn(sample_size/2);
    sample.rows(sample_size/2,sample_size-1) = 2*arma::randn(sample_size/2) + 10;

    std::string K = "gaussian"; // gaussian, square, triangle, parabolic
    std::string estimator = "grid_cv"; // rule_of_thumb, min_sd_iqr, plug_in, grid_cv
    bool enable_binning = true;
    numerics::KDE pdf_hat(K, estimator, enable_binning);
    // kde pdf_hat(0.8, K, enable_binning);
    pdf_hat.fit(sample);

    arma::vec x = arma::linspace(-5,20,1000);
    arma::vec p_hat = pdf_hat.predict(x);

    std::cout << "selected bandwidth: " << pdf_hat.bandwidth << std::endl;

    int N = 1000;
    arma::vec sample_new = pdf_hat.sample(N); // we can resample using our KDE

    int n_bins = 30;

    auto xx = arma::conv_to<dvec>::from(x);
    auto hist = arma::conv_to<dvec>::from(sample);
    auto bootstrap = arma::conv_to<dvec>::from(sample_new);
    auto p_hat0 = arma::conv_to<dvec>::from(sample_size*p_hat);

    matplotlibcpp::named_hist("sample", hist, n_bins);
    matplotlibcpp::named_plot("kde",xx, p_hat0, "-k");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    p_hat0 = arma::conv_to<dvec>::from(N*p_hat);
    matplotlibcpp::named_hist("bootstrap sample", bootstrap, n_bins);
    matplotlibcpp::named_plot("kde",xx, p_hat0, "-k");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}