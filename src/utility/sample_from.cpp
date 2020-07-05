#include "numerics.hpp"

int numerics::sample_from(const arma::vec& pdf, const arma::uvec& labels) {
    int n = pdf.n_elem;
    int i;
    double cdf = 0, rval = arma::randu();
    for (i=0; i < n; ++i) {
        if (cdf < rval && rval <= cdf + pdf(i)) break;
        cdf += pdf(i);
    }
    if ( labels.is_empty() ) return i;
    else return labels(i);
}

arma::uvec numerics::sample_from(int n, const arma::vec& pdf, const arma::uvec& labels) {
    int m = pdf.n_elem;
    arma::vec rvals = arma::randu(n);
    arma::uvec samples = arma::zeros<arma::uvec>(n);
    double cdf = 0;
    double i;
    for (i=0; i < m; ++i) {
        samples(arma::find(cdf < rvals && rvals <= cdf + pdf(i))).fill(i);
        cdf += pdf(i);
    }
    if (labels.is_empty()) return samples;
    else return labels(samples);
}