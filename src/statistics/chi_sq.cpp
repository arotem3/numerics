#include "statistics.hpp"

/* CHI_TEST : Chi squared test for homogeneity/independece.
 * --- observed : matrix of all observed values.
 * --- H1 : {HOMOGENEITY, INDEPENDENCE, GOF}. */
statistics::category_test statistics::chi_test(arma::mat& observed, hypothesis H1) {
    arma::vec row_sums = arma::sum(observed, 1);
    arma::rowvec col_sums = arma::sum(observed, 0);
    double total = arma::accu(row_sums);
    arma::mat E = col_sums * row_sums / total;
    E = arma::pow(observed - E, 2.0) / E;
    category_test test;
    test.X2 = arma::accu(E);
    test.df = observed.n_elem - 2;
    test.H1 = H1;
    test.p  = chiCDF(test.X2, test.df);
    return test;
}