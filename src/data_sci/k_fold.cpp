#include "numerics.hpp"

/* K_FOLD : returns a vector of x,y pairs of data groupings.
 * --- X : independent variable data matrix.
 * --- Y : dependent variable data matrix.
 * --- k : number of folds.
 * --- dim : dimension to split along. default dim=0, i.e. split the colomns where each row is treated as a data point. dim=1 implies each colomn is a data point and we split the rows. */
numerics::folds numerics::k_fold(const arma::mat& X, const arma::mat& Y, uint k, uint dim) {
    int m = X.n_rows, n = X.n_cols;
    arma::umat I;
    std::vector<data_pair> folds(k);
    if (dim==0) {
        arma::uvec range = arma::shuffle(arma::regspace<arma::uvec>(0,m-1));
        I = arma::reshape(range, m/k, k);
        for (uint i=0; i < k; ++i) {
            folds.at(i).X = X.rows(I.col(i));
            folds.at(i).Y = Y.rows(I.col(i));
            folds.at(i).indices = I.col(i);
            range = arma::find(arma::regspace<arma::uvec>(0,k-1) != i);
            arma::umat ii = I.cols(range);
            folds.at(i).exclude_indices = arma::vectorise(ii);
        }
    } else {
        arma::uvec range = arma::regspace<arma::uvec>(0,n-1);
        I = arma::reshape(range, k, n/k);
        for (uint i=0; i < k; ++i) {
            folds.at(i).X = X.cols(I.row(i));
            folds.at(i).Y = Y.cols(I.row(i));
            folds.at(i).indices = I.row(i);
            range = arma::find(arma::regspace<arma::uvec>(0,k-1) != i);
            arma::umat ii = I.rows(range);
            folds.at(i).exclude_indices = arma::vectorise(ii).t();
        }
    }
    return folds;
}