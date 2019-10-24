#include <numerics.hpp>

numerics::knn_regression::knn_regression(uint K, knn_algorithm algorithm, knn_metric metric) {
    k = K;
    alg = algorithm;
    metr = metric;
    categorical_loss = false;
}

numerics::knn_regression::knn_regression(const arma::uvec K_set, knn_algorithm algorithm, knn_metric metric) {
    kk = K_set;
    alg = algorithm;
    metr = metric;
    categorical_loss = false;
}

double numerics::knn_regression::fit_no_replace(const arma::mat& train_X, const arma::mat& train_Y, const arma::mat& test_X, const arma::mat& test_Y, int K) {
    arma::mat yhat(arma::size(test_Y));
    if (alg == knn_algorithm::KD_TREE) {
        numerics_private_utility::kd_tree_util::kd_tree T(train_X);
        for (int i=0; i < test_X.n_rows; ++i) {
            arma::uvec nn = T.index_kNN(test_X.row(i), K);
            yhat.row(i) = voting_func( test_X.row(i), train_X.rows(nn), train_Y.rows(nn) );
        }
    } else {
        for (int i=0; i < test_X.n_rows; ++i) {
            arma::uvec nn = brute_knn(test_X.row(i), train_X, K);
            yhat.row(i) = voting_func( test_X.row(i), train_X.rows(nn), train_Y.rows(nn) );
        }
    }
    double score = 0;
    if (categorical_loss) {
        arma::uvec ind = arma::index_max(yhat,1);
        arma::umat categories = arma::zeros<arma::umat>(arma::size(yhat));
        for (uint i=0; i < ind.n_rows; ++i) categories(i,ind(i)) = 1;
        double precision = 0, recall = 0;
        for (uint i=0; i < yhat.n_cols; ++i) {
            recall += arma::sum(categories.col(i) == test_Y.col(i) && categories.col(i)) / (double)arma::sum(test_Y.col(i));
            precision += arma::sum(categories.col(i) == test_Y.col(i) && categories.col(i)) / (double)arma::sum(categories.col(i));
        }
        score += (2*precision*recall) / (precision + recall) / yhat.n_cols;
    } else score = arma::norm(yhat - test_Y,"fro") / test_X.n_rows;

    return score;
}

numerics::knn_regression& numerics::knn_regression::fit(const arma::mat& x, const arma::mat& y) {
    if (x.n_rows != y.n_rows) {
        std::cout << "knn_regression::fit() error: number of x observations (" << x.n_rows << ") differs from the number of y observations (" << y.n_rows << ").\n";
        return *this;
    }
    Y = y;
    if (alg == knn_algorithm::AUTO) {
        if (x.n_cols < 10 && std::pow(2,x.n_cols) < x.n_rows) alg = knn_algorithm::KD_TREE;
        else alg = knn_algorithm::BRUTE;
    }
    if (!kk.is_empty()) { // cross validation
        k_folds split(x,y,3);
        cv_scores = arma::zeros(arma::size(kk));
        #pragma omp parallel
        #pragma omp for
        for (uint i=0; i < kk.n_elem; ++i) {
            double score = 0;
            for (int j=0; j < 3; ++j) {
                score += fit_no_replace(
                    split.train_set_X(j),
                    split.train_set_Y(j),
                    split.test_set_X(j),
                    split.test_set_Y(j),
                    kk(i)
                );
            }
            #pragma omp critical
            cv_scores(i) = score/3;
        }
        int ind;
        if (categorical_loss) ind = cv_scores.index_max();
        else ind = cv_scores.index_min();
        k = kk(ind);
    }

    if (alg == knn_algorithm::KD_TREE) X_tree = numerics_private_utility::kd_tree_util::kd_tree(x);
    else X_array = x;
    return *this;
}

arma::mat numerics::knn_regression::predict(const arma::mat& xgrid) {
    arma::mat yhat = arma::zeros(xgrid.n_rows, Y.n_cols);
    if (alg == knn_algorithm::KD_TREE) {
        #pragma omp parallel
        #pragma omp for
        for (int i=0; i < xgrid.n_rows; ++i) {
            arma::uvec nn = X_tree.index_kNN(xgrid.row(i), k);
            yhat.row(i) = voting_func(xgrid.row(i), X_tree.data().rows(nn), Y.rows(nn));
        }
    } else {
        #pragma omp parallel
        #pragma omp for
        for (int i=0; i < xgrid.n_rows; ++i) {
            arma::uvec nn = brute_knn(xgrid.row(i), X_array, k);
            yhat.row(i) = voting_func(xgrid.row(i), X_array.rows(nn), Y.rows(nn));
        }
    }
    return yhat;
}

arma::rowvec numerics::knn_regression::voting_func(const arma::rowvec& pt, const arma::mat& x, const arma::mat& y) {
    if (metr == knn_metric::CONSTANT) {
        return arma::mean(y);
    } else {
        arma::vec W = arma::zeros(x.n_rows);
        arma::mat d = x;
        d.each_row() -= pt;
        for (uint i=0; i < y.n_rows; ++i) {
            W(i) = arma::dot(d.row(i),d.row(i));
        }
        double s = arma::mean(W);
        W = arma::exp(-W/s);
        W /= arma::sum(W);
        return W.t() * y;
    }
}

arma::uvec numerics::knn_regression::brute_knn(const arma::rowvec& pt, const arma::mat& x, int K) {
    numerics::numerics_private_utility::kd_tree_util::pqueue Q;
    for (uint i=0; i < x.n_rows; ++i) {
        double dist = arma::norm(x.row(i) - pt);
        if (Q.size() < K || dist < Q.top().dist) {
            if (Q.size() >= K) Q.pop();
            numerics::numerics_private_utility::kd_tree_util::dist_ind DI;
            DI.dist = dist;
            DI.ind = i;
            Q.push(DI);
        }
    }
    arma::uvec inds = arma::zeros<arma::uvec>(K);
    for (int i=k-1; i >= 0; --i) {
        inds(i) = Q.top().ind;
        Q.pop();
    }
    return inds;
}

arma::mat numerics::knn_regression::get_cv_results() const {
    arma::mat rslts = arma::zeros(kk.n_elem,2);
    for (uint i=0; i < kk.n_elem; ++i) rslts(i,0) = kk(i);
    rslts.col(1) = cv_scores;
    return rslts;
}