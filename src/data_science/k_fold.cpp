#include <numerics.hpp>

/* k_folds(x, y, k, dim) : split data into k many groups, i.e. folds.
 * --- X : independent variable data matrix.
 * --- Y : dependent variable data matrix.
 * --- k : number of folds.
 * --- dim : dimension to split along. default dim=0, i.e. split the colomns where each row is treated as a data point. dim=1 implies each colomn is a data point and we split the rows. */
numerics::k_folds::k_folds(const arma::mat& x, const arma::mat& y, uint k, uint dim) {
    int m = x.n_rows, n = x.n_cols;
    direction = dim;
    num_folds = k;

    X = x;
    Y = y;
    if (direction==0) {
        range = arma::shuffle(arma::regspace<arma::uvec>(0,m-1));
        I = arma::reshape(range, m/k, k);
    } else {
        range = arma::shuffle(arma::regspace<arma::uvec>(0,n-1));
        I = arma::reshape(range, n/k, k);
    }
    range = arma::regspace<arma::uvec>(0,k-1);
}

/* fold_X(j) : return the x-values of the j^th fold */
arma::mat numerics::k_folds::fold_X(uint j) {
    if (j >= num_folds) {
        std::cerr << "k_folds element access index (=" << j << ") is out of range of the size - 1 (=" << num_folds << ")." << std::endl;
        return arma::mat();
    }
    if (direction==0) return X.rows( I.col(j) );
    else return X.cols( I.col(j) );
}

/* fold_Y(j) : return the y-values of the j^th fold */
arma::mat numerics::k_folds::fold_Y(uint j) {
    if (j >= num_folds) {
        std::cerr << "k_folds element access index (=" << j << ") is out of range of the size - 1 (=" << num_folds << ")." << std::endl;
        return arma::mat();
    }
    if (direction==0) return Y.rows( I.col(j) );
    else return Y.cols( I.col(j) );
}

/* not_fold_X(j) : return the x-values of all but j^th fold */
arma::mat numerics::k_folds::not_fold_X(uint j) {
    if (j >= num_folds) {
        std::cerr << "k_folds element access index (=" << j << ") is out of range of the size - 1 (=" << num_folds << ")." << std::endl;
        return arma::mat();
    }
    arma::umat ii = I.cols(arma::find(range != j));
    ii = arma::vectorise(ii);
    if (direction==0) return X.rows(ii);
    else return X.cols(ii);
}

/* not_fold_Y(j) : return the y-values of all but j^th fold */
arma::mat numerics::k_folds::not_fold_Y(uint j) {
    if (j >= num_folds) {
        std::cerr << "k_folds element access index (=" << j << ") is out of range of the size - 1 (=" << num_folds << ")." << std::endl;
        return arma::mat();
    }
    arma::umat ii = I.cols(arma::find(range != j));
    ii = arma::vectorise(ii);
    if (direction==0) return Y.rows(ii);
    else return Y.cols(ii);
}

/* k_folds::[j] : return the x-values of the j^th fold, if j is negative, returns the x-values of all but |j| - 1 fold. */
arma::mat numerics::k_folds::operator[](int k) {
    if (k < 0) return not_fold_X(-1-k);
    else return fold_X(k); 
}

/* k_folds::(j) : return the y-values of the j^th fold, if j is negative, returns the y-values of all but |j| - 1 fold. */
arma::mat numerics::k_folds::operator()(int k) {
    if (k < 0) return not_fold_Y(-1-k);
    else return fold_Y(k);
}