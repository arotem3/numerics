#include <numerics.hpp>

void numerics::KMeans::_check_k(int k) {
    if (k < 2) {
        throw std::invalid_argument("require number of clusters k must be > 1 (k provided was " + std::to_string(k) + ").");
    }
}

void numerics::KMeans::_check_maxiter(long maxiter) {
    if (_max_iter < 1) {
        throw std::invalid_argument("require _max_iter (=" + std::to_string(_max_iter) + ") >= 1");
    }
}

void numerics::KMeans::_check_tol(double tol) {
    if (tol < 0) {
        throw std::invalid_argument("require _tol (=" + std::to_string(tol) + ") >= 0.0");
    }
}

/* _d(a,b) : __private__ compute the distance function between a and b */
double numerics::KMeans::_d(const arma::rowvec& a, const arma::rowvec& b) const {
    if (_p == 0) {
        return arma::norm(a - b, "inf");
    } else {
        return arma::norm(a - b, _p);
    }
}

/* _update_labels(label, nearest_dist, sum_p, data, m): __private__ update labels, nearest_dist, and sum_p using the first m<=k cluster centers */
void numerics::KMeans::_update_labels(arma::uvec& label, arma::vec& nearest_dist, double& sum_p, const arma::mat& data, uint m) {
    uint n = data.n_rows;
    for (uint j=0; j < n; ++j) {
        uint minC = label(j);
        double minDist = nearest_dist(j);
        for (uint l=0; l < m; ++l) {
            if ((_intra_dist(l,minC) >= 2*minDist) || (l==minC)) continue;
            
            double dist = _d(data.row(j), _clusters.row(l));

            if (dist < minDist) {
                minDist = dist;
                minC = l;
            }
        }
        label(j) = minC;
        sum_p += minDist - nearest_dist(j); // we are replacing nearest_dist(j) by minDist so we need to account for that in the sum
        nearest_dist(j) = minDist;
    }
}

/* _update_labels(labels, data) : __private__ update labels without precomputed distances, this is used between iterations.  */
void numerics::KMeans::_update_labels(arma::uvec& labels, const arma::mat& data) const {
    uint n = data.n_rows;
    for (uint j=0; j < n; ++j) {
        uint minC = labels(j);
        double minDist = _d(data.row(j), _clusters.row(minC));
        for (uint l=0; l < _k; ++l) {
            if ((_intra_dist(l,minC) >= 2*minDist) || (l==minC)) continue;
            
            double dist = _d(data.row(j), _clusters.row(l));

            if (dist < minDist) {
                minDist = dist;
                minC = l;
            }
        }
        labels(j) = minC;
    }
}

/* _update_intra_dist(): __private__ compute the distances between clusters. */
void numerics::KMeans::_update_intra_dist() {
    for (uint i=1; i < _k; ++i) {
        for (uint j=0; j < i; ++j) {
            _intra_dist(i,j) = _d(_clusters.row(i), _clusters.row(j));
            _intra_dist(j,i) = _intra_dist(i,j); // enforce symmetry
        }
    }
}

/* init_clusters(label, data): __private__ initialize clusters and more information with KMeans++ improved by triangle inequality speed-up */
void numerics::KMeans::_init_clusters(arma::uvec& label, const arma::mat& data) {
    uint n = data.n_rows; uint dim = data.n_cols;
    _clusters = arma::zeros(_k, dim);
    
    uint idx = std::rand() % n; // initialize first cluster randomly from the data
    _clusters.row(0) = data.row(idx);
    
    label = arma::zeros<arma::uvec>(n); // label all the data as belonging to the first cluster
    if (_k <= 1) return;

    arma::vec nearest_dist = arma::zeros(n); // nearest_dist(j) is the distance of the j^th observation from the nearest cluster center
    double sum_p = 0;
    for (uint j=0; j < n; ++j) {
        if (j == idx) continue;
        else {
            double dist = _d(data.row(j), _clusters.row(0));
            sum_p += dist;
            nearest_dist(j) = dist;
        }
    }
    idx = numerics::sample_from(nearest_dist/sum_p);
    _clusters.row(1) = data.row(idx);
    if (_k <= 2) return;
    
    _intra_dist = arma::zeros(_k,_k);
    _intra_dist(0,1) = nearest_dist(idx);
    _intra_dist(1,0) = _intra_dist(0,1);
    
    for (uint m=2; m < _k; ++m) {
        _update_labels(label, nearest_dist, sum_p, data, m);
        idx = numerics::sample_from(nearest_dist/sum_p);
        _clusters.row(m) = data.row(idx);

        for (uint j=0; j < m; ++j) {
            _intra_dist(m,j) = _d(_clusters.row(m), _clusters.row(j));
            _intra_dist(j,m) = _intra_dist(m,j);
        }
    }
    _update_labels(label, nearest_dist, sum_p, data, _k); // update labels again to acount for the last added cluster
}

void numerics::KMeans::fit(const arma::mat& data) {
    fit_predict(data);
}

arma::uvec numerics::KMeans::fit_predict(const arma::mat& data) {
    if (data.n_rows < _k) {
        std::string excpt = "KMeans::fit() error: number of observations (data.n_rows = " + std::to_string(data.n_rows) + ") is less than the requested number of clusters (k = " + std::to_string(_k) + ").";
        throw std::logic_error(excpt);
    }
    _dim = data.n_cols;
    arma::uvec labels;
    _init_clusters(labels, data);
    
    uint n = data.n_rows;
    uint iter=0;
    while (true) {
        arma::vec update_diff = arma::zeros(_k);
        for (uint i=0; i < _k; ++i) {
            arma::uvec idx = arma::find(labels == i);
            arma::rowvec ci = arma::mean(data.rows(idx), 0);

            if (_tol > 0) update_diff(i) += arma::norm(ci - _clusters.row(i), "inf");

            _clusters.row(i) = ci;
        }

        if (_tol > 0 && update_diff.max() < _tol) break;

        iter++;
        if (iter > _max_iter) {
            std::cerr << "KMeans() warnings: failed to converge within the maximum number of iterations allowed.\n";
            break;
        }
        _update_intra_dist();
        _update_labels(labels, data);
    }
    _update_intra_dist();
    _update_labels_nearest(labels, data);
    return labels;
}

/* predict(nearest_dist, data) : predict labels for new data and compute the distance between each data point and the nearest cluster. */
arma::uvec numerics::KMeans::predict(const arma::mat& data) const {
    _check_x(data);
    uint n = data.n_rows;
    arma::uvec labels = arma::zeros<arma::uvec>(n);
    _update_labels(labels, data);
    return labels;
}

/* _update_batch_labels(labels, data, p, i, f) : update labels in the range [i, f) of p */
void numerics::KMeansSGD::_update_batch_labels(arma::uvec& labels, const arma::mat& data, const arma::uvec& p, uint i, uint f) {
    for (uint j=i; j < f; ++j) {
        uint minC = labels(p(j));
        double minDist = _d(data.row(p(j)), _clusters.row(minC));
        for (uint l=0; l < _k; ++l) {
            if ((_intra_dist(l,minC) >= 2*minDist) || (l==minC)) continue;
            
            double dist = _d(data.row(p(j)), _clusters.row(l));

            if (dist < minDist) {
                minDist = dist;
                minC = l;
            }
        }
        labels(p(j)) = minC;
    }
}

/* _update_labels_nearest(labels, data) : update labels while simultaneously compute nearest points */
void numerics::KMeans::_update_labels_nearest(arma::uvec& labels, const arma::mat& data) {
    uint n = data.n_rows;
    arma::vec nearest_dist(_k);
    for (uint i=0; i < _k; ++i) {
        nearest_dist(i) = _d(data.row(0), _clusters.row(i));
    }
    labels(0) = nearest_dist.index_min();

    _nearest_idx = arma::zeros<arma::uvec>(_k);

    for (uint j=1; j < n; ++j) {
        uint minC = labels(j);
        double minDist = _d(data.row(j), _clusters.row(minC));
        for (uint l=0; l < _k; ++l) {
            if ((_intra_dist(l,minC) >= 2*minDist) || (l==minC)) continue;
            
            double dist = _d(data.row(j), _clusters.row(l));

            if (dist < minDist) {
                minDist = dist;
                minC = l;
            }
        }
        labels(j) = minC;
        if (minDist < nearest_dist(labels(j))) {
            _nearest_idx(labels(j)) = j;
            nearest_dist(labels(j)) = minDist;
        }
    }
    _nearest_point = data.rows(_nearest_idx);
}

void numerics::KMeansSGD::_sgd_steps(arma::uvec& labels, const arma::mat& data) {
    u_long n = data.n_rows;
    arma::uvec counts = arma::ones<arma::uvec>(_k); // the original algorithm calls for counts = zeros(_k) but since we are using KMeans++ we already have initialized data with one member
    arma::uvec p = arma::randperm<arma::uvec>(n);
    uint b = 0;
    for (uint i=0; i < _max_iter; ++i) {
        uint minbbn = std::min(b+_batch_size, n);
        _update_intra_dist(); // update distances and labels before each batch run
        _update_batch_labels(labels, data, p, b, minbbn);
        for (uint j=0; j < _batch_size; ++j) {
            uint c = labels(p(b));
            counts(c)++;
            long double eta = 1.0l / counts(c);
            _clusters.row(c) = (1.0l - eta)*_clusters.row(c) + eta*data.row(p(b));
            b = (b+1) % n;
        }
    }
    _update_intra_dist();
    _update_labels_nearest(labels, data);
}

/* fit(data, _batch_size=100, _max_iter=10) : initialize and fit the data using batch gradient descent.
 * --- data : training data set. */
arma::uvec numerics::KMeansSGD::fit_predict(const arma::mat& data) {
    if (data.n_rows < _k) {
        std::string excpt = "KMeans::fit() error: number of observations (data.n_rows = " + std::to_string(data.n_rows) + ") is less than the requested number of clusters (k = " + std::to_string(_k) + ").";
        throw std::logic_error(excpt);
    }
    _dim = data.n_cols;
    arma::uvec labels;
    _init_clusters(labels, data);

    _sgd_steps(labels, data);
    return labels;
}

void numerics::KMeansSGD::fit(const arma::mat& data) {
    fit_predict(data);
}

void numerics::KMeansSGD::_check_batch(long b) {
    if (b < 1) {
        throw std::invalid_argument("require batch_size (=" + std::to_string(b) + ") >= 1");
    }
}