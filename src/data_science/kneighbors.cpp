#include <numerics.hpp>

// void numerics::KNeighborsRegressor::fit(const arma::mat& X, const arma::vec& y) {   
//     if (not _ks.is_empty()) {
//         k_folds split(X, y, 3);

//         _scores = arma::zeros(_ks.n_elem);

//         for (int i=0; i < 3; ++i) {
//             KNeighborsRegressor knn(_ks(i), _weighted, _data.p, _data.leaf_size);
//             knn.fit(split.train_set_X(i), split.train_set_Y(i));
//             _scores(i) += knn.score(split.test_set_X(i), split.test_set_Y(i));
//         }
//         _scores /= 3;
//         int best_score = _scores.index_max();
//         _k = _ks(best_score);
//     }
//     _data.fit(X);
//     _y = y;
// }

// double numerics::KNeighborsRegressor::score(const arma::mat& X, const arma::vec& y) const {
//     return r2_score(y, predict(X));
// }

// arma::vec numerics::KNeighborsRegressor::predict(const arma::mat& X) const {
//     arma::vec yh; yh.resize(X.n_rows);
//     arma::umat indices;
//     if (_weighted) {
//         _data.query(indices, X, _k);

//         for (u_long i=0; i < X.n_rows; ++i) {
//             yh(i) = arma::mean(_y(indices.row(i)));
//         }
//     } else {
//         arma::mat distances;
//         _data.query(distances, indices, X, _k);
        
//         distances.each_row(
//             [](arma::rowvec& u) -> void {
//                 u += 1.0e-8; // prevent zero division.
//                 u = 1 / u;
//                 u /= arma::sum(u);
//             }
//         );

//         for (u_long i=0; i < X.n_rows; ++i) {
//             yh(i) = arma::dot(_y(indices.row(i)), distances.row(i));
//         }
//     }
//     return yh;
// }
 
// void numerics::KNeighborsClassifier::fit(const arma::mat& X, const arma::uvec& y) {
//     if (not _ks.is_empty()) {
//         KFolds<double,arma::uword> split(3);

//         _scores = arma::zeros(_ks.n_elem);

//         for (int i=0; i < 3; ++i) {
//             KNeighborsClassifier knn(_ks(i), _weights, _data.p, _data.leaf_size);
//             knn.fit(split.trainX(i), split.trainY(i));
//             _scores(i) += knn.score(split.test_set_X(i), split.test_Y(i));
//         }
//         scores /= 3;
//         int best_score = _scores.index_max();
//         _k = _ks(best_score);
//     }
// }
// /* knn_regression(K, algorithm=AUTO, metric=CONSTANT) : initialize k-neighbors regression object, by specifying k.
//  * --- K : number of neighbors to use during prediction.
//  * --- algorithm : {AUTO,BRUTE,KD_TREE} algorithm for finding nearest neighbors. BRUTE O(n) nearest neighbor query time. KD_TREE O(log n) nearest neighbor query time; less ideal for high dimensional data.
//  * --- metric : {CONSTANT,L2_DISTANCE,L1_DISTANCE} apply weights to neighbors estimates. */
// numerics::knn_regression::knn_regression(uint K, numerics::knn_algorithm algorithm, numerics::knn_metric metric) : num_neighbors(k), data_X(X_array), tree_X(X_tree), data_Y(Y) {
//     k = K;
//     alg = algorithm;
//     metr = metric;
//     categorical_loss = false;
// }

// /* knn_classifier(K, algorithm=AUTO, metric=CONSTANT) : nitialize k-neighbors classification object, by specifying k.
//  * --- K : number of neighbors to use during prediction.
//  * --- algorithm : {AUTO,BRUTE,KD_TREE} algorithm for finding nearest neighbors. BRUTE O(n) nearest neighbor query time. KD_TREE O(log n) nearest neighbor query time; less ideal for high dimensional data.
//  * --- metric : {CONSTANT,L2_DISTANCE,L1_DISTANCE} apply weights to neighbors estimates. */
// numerics::knn_classifier::knn_classifier(uint K, numerics::knn_algorithm algorithm, numerics::knn_metric metric) : knn_regression::knn_regression(K,algorithm,metric), categories(cats) {
//     categorical_loss = true;
// }

// /* knn_regression(k_set, algorithm=AUTO, metric=CONSTANT) : initialize k-neighbors regression object, requesting cross-validation.
//  * --- k_set : a set of K's to test in cross-validation, the one minimizing testing RMSE is selected.
//  * --- algorithm : {AUTO,BRUTE,KD_TREE} algorithm for finding nearest neighbors. BRUTE O(n) nearest neighbor query time. KD_TREE O(log n) nearest neighbor query time; less ideal for high dimensional data.
//  * --- metric : {CONSTANT,L2_DISTANCE,L1_DISTANCE} apply weights to neighbors estimates. */
// numerics::knn_regression::knn_regression(const arma::uvec K_set, knn_algorithm algorithm, knn_metric metric) : num_neighbors(k), data_X(X_array), tree_X(X_tree), data_Y(Y) {
//     alg = algorithm;
//     metr = metric;
//     categorical_loss = false;

//     if (arma::any(K_set <= 0)) {
//         std::cerr << "knn_regression error: all K_set must be >=0. (K_set provided contains :" << K_set(arma::find(K_set <= 0)).t() << ").\n";
//         return;
//     }
//     kk = K_set;
// }

// /* knn_classifier(k_set, algorithm=AUTO, metric=CONSTANT) : initialize k-neighbors classifier object, requesting cross-validation.
//  * --- k_set : a set of K's to test in cross-validation, the one minimizing testing F1 is selected.
//  * --- algorithm : {AUTO,BRUTE,KD_TREE} algorithm for finding nearest neighbors. BRUTE O(n) nearest neighbor query time. KD_TREE O(log n) nearest neighbor query time; less ideal for high dimensional data.
//  * --- metric : {CONSTANT,L2_DISTANCE,L1_DISTANCE} apply weights to neighbors estimates. */
// numerics::knn_classifier::knn_classifier(const arma::uvec K_set, knn_algorithm algorithm, knn_metric metric) : knn_regression::knn_regression(K_set,algorithm,metric), categories(cats) {
//     categorical_loss = true;
// }

// /* score_regression(train_X, train_Y, test_X, test_Y, K) : __private__ returns RMSE for regression and F1 score for classifier.
//  * --- train_X, train_Y : training data
//  * --- test_X, test_Y : testing data
//  * --- number of neighbors */
// double numerics::knn_regression::score_regression(const arma::mat& train_X, const arma::mat& train_Y, const arma::mat& test_X, const arma::mat& test_Y, int K) {
//     arma::mat yhat(arma::size(test_Y));
//     if (alg == knn_algorithm::KD_TREE) {
//         numerics_private_utility::kd_tree_util::kd_tree T(train_X);
//         for (int i=0; i < test_X.n_rows; ++i) {
//             arma::uvec nn = T.index_kNN(test_X.row(i), K);
//             yhat.row(i) = voting_func( test_X.row(i), train_X.rows(nn), train_Y.rows(nn) );
//         }
//     } else {
//         for (int i=0; i < test_X.n_rows; ++i) {
//             arma::uvec nn = brute_knn(test_X.row(i), train_X, K);
//             yhat.row(i) = voting_func( test_X.row(i), train_X.rows(nn), train_Y.rows(nn) );
//         }
//     }

//     double score = 0;
    
//     if (categorical_loss) { // macro-F1 score
//         arma::uvec ind = arma::index_max(yhat,1);
//         arma::umat categories = arma::zeros<arma::umat>(arma::size(yhat));
//         for (uint i=0; i < ind.n_rows; ++i) categories(i,ind(i)) = 1;
//         double precision = 0, recall = 0;
//         for (uint i=0; i < yhat.n_cols; ++i) {
//             recall += arma::sum(categories.col(i) == test_Y.col(i) && categories.col(i)) / (double)arma::sum(test_Y.col(i));
//             precision += arma::sum(categories.col(i) == test_Y.col(i) && categories.col(i)) / (double)arma::sum(categories.col(i));
//         }
//         score += (2*precision*recall) / (precision + recall) / yhat.n_cols;
//     } else score = arma::norm(yhat - test_Y,"fro") / test_X.n_rows; // RMSE

//     return score;
// }

// /* fit(x, y) : fit knn regressor, and perform cross-validation if applicable.
//  * --- x : independent variable.
//  * --- y : dependent variable. */
// void numerics::knn_regression::fit(const arma::mat& x, const arma::mat& y) {
//     if (x.n_rows != y.n_rows) {
//         std::cout << "knn_regression::fit() error: number of x observations (" << x.n_rows << ") differs from the number of y observations (" << y.n_rows << ").\n";
//         return;
//     }

//     Y = y;

//     if (alg == knn_algorithm::AUTO) {
//         if (x.n_cols < 10 && std::pow(2,x.n_cols) < x.n_rows) alg = knn_algorithm::KD_TREE;
//         else alg = knn_algorithm::BRUTE;
//     }

//     if (!kk.is_empty()) { // cross validation
//         k_folds split(x,y,3);
//         arma::vec cv_scores = arma::zeros(arma::size(kk));
//         #pragma omp parallel for
//         for (uint i=0; i < kk.n_elem; ++i) {
//             double score = 0;
//             for (int j=0; j < 3; ++j) {
//                 score += score_regression(
//                     split.train_set_X(j),
//                     split.train_set_Y(j),
//                     split.test_set_X(j),
//                     split.test_set_Y(j),
//                     kk(i)
//                 );
//             }
//             #pragma omp critical
//             cv_scores(i) = score/3;
//         }
//         int ind;
//         if (categorical_loss) ind = cv_scores.index_max();
//         else ind = cv_scores.index_min();
//         k = kk(ind);
//     }

//     if (alg == knn_algorithm::KD_TREE) X_tree = numerics_private_utility::kd_tree_util::kd_tree(x);
//     else X_array = x;
// }

// /* fit(x, y) : fit knn classifier, provided categorical data.
//  * --- x : independent variable.
//  * --- y : class labels, need not be 0,...,m-1; the unique elements will be extracted. */
// void numerics::knn_classifier::fit(const arma::mat& X, const arma::uvec& y) {
//     cats = arma::unique<arma::uvec>(y);
//     int n_categories = categories.n_elem;
//     Y = arma::zeros(X.n_rows, n_categories);
//     for (int i=0; i < n_categories; ++i) {
//         arma::uvec cat_i = arma::find(y == categories(i));
//         for (arma::uword j : cat_i) {
//             Y(j,i) = 1;
//         }
//     }
//     knn_regression::fit(X,Y);
// }

// /* predict(xgrid) : predict regressor
//  * --- xgrid : points to predict over */
// arma::mat numerics::knn_regression::predict(const arma::mat& xgrid) {
//     arma::mat yhat = arma::zeros(xgrid.n_rows, Y.n_cols);
//     if (alg == knn_algorithm::KD_TREE) {
//         #pragma omp parallel
//         #pragma omp for
//         for (int i=0; i < xgrid.n_rows; ++i) {
//             arma::uvec nn = X_tree.index_kNN(xgrid.row(i), k);
//             yhat.row(i) = voting_func(xgrid.row(i), X_tree.data().rows(nn), Y.rows(nn));
//         }
//     } else {
//         #pragma omp parallel
//         #pragma omp for
//         for (int i=0; i < xgrid.n_rows; ++i) {
//             arma::uvec nn = brute_knn(xgrid.row(i), X_array, k);
//             yhat.row(i) = voting_func(xgrid.row(i), X_array.rows(nn), Y.rows(nn));
//         }
//     }
//     return yhat;
// }

// /* operator()(xgrid) : predict regressor, same as predict(xgrid).
//  * --- xgrid : points to predict over. */
// arma::mat numerics::knn_regression::operator()(const arma::mat& xgrid) {
//     return predict(xgrid);
// }

// /* predict_probabilities(xgrid) : predict the probability associated with each category. Returns a matrix whose rows sum to 1.
//  * --- xgrid : query points to predict probability vectors for. */
// arma::mat numerics::knn_classifier::predict_probabilities(const arma::mat& xgrid) {
//     return predict(xgrid);
// }

// /* predict_categories(xgrid) : predict categories indexed from the categorical data given during fit.
//  * --- xgrid : query points to predict categories for. */
// arma::uvec numerics::knn_classifier::predict_categories(const arma::mat& xgrid) {
//     if (cats.is_empty()) cats = arma::regspace<arma::uvec>(0,Y.n_cols);
//     arma::uvec ind = arma::index_max(predict_probabilities(xgrid),1);
//     arma::uvec cHat = cats(ind);
//     return cHat;
// }

// /* voting_func(pt, x, y) : __private__ compute (weighted) average of points (x,y) relative to query point pt.
//  * --- pt : query point.
//  * --- x : independent variable, used to compute weights if metric != CONSTANT.
//  * --- y : dependent variable, values to compute average of. */
// arma::rowvec numerics::knn_regression::voting_func(const arma::rowvec& pt, const arma::mat& x, const arma::mat& y) {
//     arma::vec W = arma::ones(y.n_cols);
//     if (metr == knn_metric::CONSTANT) {
//         return arma::mean(y);
//     } else if (metr == knn_metric::L1_DISTANCE) {
//         W = arma::sum(arma::abs(x.each_row() - pt), 1);
//     } else { // if (metr == knn_metric::L2_DISTANCE)
//         W = arma::sum(arma::square(x.each_row() - pt), 1);
//     }
//     if (W.has_nan()) {
//         arma::uvec nan_idx = arma::find_nonfinite(W);
//         W.fill(0);
//         W(arma::find(nan_idx)).fill(1);
//     }
//     W = arma::sum(W) - W;
//     W = arma::exp(W-W.max());
//     W = W / arma::sum(W);
//     return W.t() * y;
// }

// /* brute_knn(pt, x, K) : __private__ computes the indices of the k-neighbors of query point pt.
//  * --- pt : query point.
//  * --- x : data from which to find nearest neighbors.
//  * --- K : number of neighbors to find. */
// arma::uvec numerics::knn_regression::brute_knn(const arma::rowvec& pt, const arma::mat& x, int K) {
//     numerics::numerics_private_utility::kd_tree_util::pqueue Q;
//     for (uint i=0; i < x.n_rows; ++i) {
//         double dist = arma::norm(x.row(i) - pt);
//         if (Q.size() < K || dist < Q.top().dist) {
//             if (Q.size() >= K) Q.pop();
//             numerics::numerics_private_utility::kd_tree_util::dist_ind DI;
//             DI.dist = dist;
//             DI.ind = i;
//             Q.push(DI);
//         }
//     }
//     K = std::min((size_t)K, Q.size());
//     arma::uvec inds = arma::zeros<arma::uvec>(K);
//     for (int i=K-1; i >= 0; --i) {
//         inds(i) = Q.top().ind;
//         Q.pop();
//     }
//     return inds;
// }