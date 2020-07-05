#include <numerics.hpp>

/* _d(a,b) : __private__ compute the distance function between a and b */
double numerics::kmeans::_d(const arma::rowvec& a, const arma::rowvec& b) {
    if (std::isinf(_p)) {
        return arma::norm(a - b, "inf");
    } else {
        return arma::norm(a - b, _p);
    }
}

/* kmeans(k, p_norm=2, max_iter=100) : initialize kmeans object.
 * --- k : number of clusters to compute.
 * --- p_norm : norm for computing differences, p_norm >= 1 or is_inf(p_norm).
 * --- max_iter : maximum number of iterations before premature stopping. */
numerics::kmeans::kmeans(uint k, uint p_norm, uint max_iter) : clusters(_clusters), cluster_distances(_intra_dist), points_nearest_centers(_nearest_point), index_nearest_centers(_nearest_idx) {
    if (k < 2) {
        std::string excpt = "kmeans error: number of clusters k must be > 1 (k provided was " + std::to_string(k) + ").";
        throw std::runtime_error(excpt);
    }
    _k = k;
    if (p_norm >= 1 || std::isinf(p_norm)) {
        _p = p_norm;
    } else {
        std::string excpt = "kmeans error: p_norm must be >=1 or inf (p_norm provided was " + std::to_string(p_norm) + ").";
        throw std::runtime_error(excpt);
    }
    _max_iter = max_iter;
}

/* _update_labels(label, nearest_dist, sum_p, data, m): __private__ update labels, nearest_dist, and sum_p using the first m<=k cluster centers */
void numerics::kmeans::_update_labels(arma::uvec& label, arma::vec& nearest_dist, double& sum_p, const arma::mat& data, uint m) {
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
void numerics::kmeans::_update_labels(arma::uvec& labels, const arma::mat& data) {
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
void numerics::kmeans::_update_intra_dist() {
    for (uint i=1; i < _k; ++i) {
        for (uint j=0; j < i; ++j) {
            _intra_dist(i,j) = _d(_clusters.row(i), _clusters.row(j));
            _intra_dist(j,i) = _intra_dist(i,j); // enforce symmetry
        }
    }
}

/* init_clusters(label, data): __private__ initialize clusters and more information with kmeans++ improved by triangle inequality speed-up */
void numerics::kmeans::_init_clusters(arma::uvec& label, const arma::mat& data) {
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

/* fit(labels, nearest_dist, data, tol=1e-2): fit kmeans object. returns the associated labels predicted from the data (one of [0, ..., k-1])
 * --- data : matrix of data (num observations X num features)
 * --- tol : tolerance for stopping criteria, this is the maximum difference between iterations, i.e. max_{i,j} |c_{i,j}^n - c_{i,j}^{n+1}|, if tol <= 0, then update checks will not be computed and instead max_iter will be used as the only stopping criteria. */
arma::uvec numerics::kmeans::fit(const arma::mat& data, double tol) {
    if (data.n_rows < _k) {
        std::string excpt = "kmeans::fit() error: number of observations (data.n_rows = " + std::to_string(data.n_rows) + ") is less than the requested number of clusters (k = " + std::to_string(_k) + ").";
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

            if (tol > 0) update_diff(i) += arma::norm(ci - _clusters.row(i), "inf");

            _clusters.row(i) = ci;
        }

        if (tol > 0 && update_diff.max() < tol) break;

        iter++;
        if (iter > _max_iter) {
            std::cerr << "kmeans() warnings: failed to converge within the maximum number of iterations allowed.\n";
            break;
        }
        _update_intra_dist();
        _update_labels(labels, data);
    }
    _update_intra_dist();
    _update_labels_nearest(labels, data);
    return labels;
}

/* predict(nearest_dist, data) : predict labels for new data and compute the distance between each data point and the nearest cluster.
 * --- nearest_dist : the i^th element is the distance between the i^th observation and the nearest cluster center. */
arma::uvec numerics::kmeans::predict(const arma::mat& data) {
    if (data.n_cols != _dim) {
        std::string excpt = "kmeans::predict() error: dimension mismatch, (data.n_cols = " + std::to_string(data.n_cols) + ") which does not equal the dimension of the fitted data (clusters.n_cols = " + std::to_string(_clusters.n_cols) + ").";
        throw std::logic_error(excpt);
    }
    uint n = data.n_rows;
    arma::uvec labels = arma::zeros<arma::uvec>(n);
    _update_labels(labels, data);
    return labels;
}

/* operator()(data) : same as predict(data). */
arma::uvec numerics::kmeans::operator()(const arma::mat& data) {
    return predict(data);
}

/* kmeans_sgd(k, p_norm=2) : initialize kmeans object computed by batch gradient descent. */
numerics::kmeans_sgd::kmeans_sgd(uint k, uint p_norm) : kmeans(k, p_norm) {}


/* _update_batch_labels(labels, data, p, i, f) : update labels in the range [i, f) of p */
void numerics::kmeans_sgd::_update_batch_labels(arma::uvec& labels, const arma::mat& data, const arma::uvec& p, uint i, uint f) {
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
void numerics::kmeans::_update_labels_nearest(arma::uvec& labels, const arma::mat& data) {
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

void numerics::kmeans_sgd::_sgd_steps(arma::uvec& labels, const arma::mat& data, uint batch_size, uint max_iter) {
    uint n = data.n_rows;
    arma::uvec counts = arma::ones<arma::uvec>(_k); // the original algorithm calls for counts = zeros(_k) but since we are using kmeans++ we already have initialized data with one member
    arma::uvec p = arma::randperm<arma::uvec>(n);
    uint b = 0;
    for (uint i=0; i < max_iter; ++i) {
        uint minbbn = std::min(b+batch_size, n);
        _update_intra_dist(); // update distances and labels before each batch run
        _update_batch_labels(labels, data, p, b, minbbn);
        for (uint j=0; j < batch_size; ++j) {
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

/* fit(data, batch_size=100, max_iter=10) : initialize and fit the data using batch gradient descent.
 * --- data : training data set.
 * --- batch_size : size of batch to use in each iteration
 * --- max_iter : total number of iteration. */
arma::uvec numerics::kmeans_sgd::fit(const arma::mat& data, uint batch_size, uint max_iter) {
    if (data.n_rows < _k) {
        std::string excpt = "kmeans::fit() error: number of observations (data.n_rows = " + std::to_string(data.n_rows) + ") is less than the requested number of clusters (k = " + std::to_string(_k) + ").";
        throw std::logic_error(excpt);
    }
    _dim = data.n_cols;
    arma::uvec labels;
    _init_clusters(labels, data);

    _sgd_steps(labels, data, batch_size, max_iter);
    return labels;
}