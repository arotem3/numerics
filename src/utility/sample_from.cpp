#include "numerics.hpp"

double numerics::sample_from(const arma::vec& pdf, const arma::vec& labels) {
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

arma::vec numerics::sample_from(int n, const arma::vec& pdf, const arma::vec& labels) {
    arma::vec rvals = arma::zeros(n);
    for (int i=0; i < n; ++i) {
        rvals(i) = sample_from(pdf, labels);
    }
    return rvals;
}