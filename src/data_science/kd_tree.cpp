#include <numerics.hpp>

void numerics::neighbors::KDTree::_valid_leaf_size(int leaf_size) {
    if (leaf_size < 1) {
        throw std::invalid_argument("require leaf_size (=" + std::to_string(leaf_size) + ") >= 1.");
    }
}

void numerics::neighbors::KDTree::_valid_p_norm(int pnorm) {
    if (pnorm < 0) {
        throw std::invalid_argument("require p_norm (=" + std::to_string(pnorm) + ") >= 0 or \"inf\" (0 indicates \"inf\").");
    }
}

double numerics::neighbors::KDTree::_distance(const arma::rowvec& a, const arma::rowvec& b) const {
    if (_p == 0) return arma::norm(a - b, "inf");
    else if (_p == 1) return arma::norm(a - b, _p);
    else return std::pow(arma::norm(a - b, _p), _p);
}

u_long numerics::neighbors::KDTree::_index_median(const std::vector<u_long>& rows, u_long col) {
    std::vector<_DistanceIndex> y;
    for (u_long i : rows) {
        y.push_back({_data(i, col), i});
    }
    int nhalf = y.size()/2;
    std::nth_element(y.begin(), y.begin()+nhalf, y.end());
    return y.at(nhalf).index;
}

void numerics::neighbors::KDTree::_find_rel(std::vector<u_long>& less, std::vector<u_long>& greater, const std::vector<u_long>& rows, u_long col, u_long pivot) {
    for (u_long i : rows) {
        if (i == pivot) continue;
        if (_data(i,col) < _data(pivot, col)) less.push_back(i);
        else greater.push_back(i);
    }
}

void numerics::neighbors::KDTree::_build_tree() {
    _clear_tree(_head);
    _head = new _Node();
    _head->dim = arma::var(_data, 0).index_max();

    std::vector<u_long> idx;
    for (u_long i=0; i < _size; ++i) idx.push_back(i);

    std::queue< std::pair<_Node*, std::vector<u_long>> > q;
    q.push({_head, idx});
    while (not q.empty()) {
        std::pair<_Node*, std::vector<u_long>> t = q.front();
        q.pop();

        if (t.second.empty()) continue;
        if (t.second.size() <= _leaf_size) {
            t.first->index = std::move(t.second);
        } else {
            u_long median_i = _index_median(t.second, t.first->dim);
            t.first->index.push_back(median_i);

            u_long next_dim = _next_dim(t.first->dim);
            std::vector<u_long> less, greater;
            _find_rel(less, greater, t.second, t.first->dim, median_i);

            if (not less.empty()) {
                t.first->left = new _Node();
                t.first->left->dim = next_dim;
                q.push({t.first->left, less});
            }
            if (not greater.empty()) {
                t.first->right = new _Node();
                t.first->right->dim = next_dim;
                q.push({t.first->right, greater});
            }
        }
    }
}

u_long numerics::neighbors::KDTree::_next_dim(u_long d) {
    return (d + 1) % _n_dims;
}

void numerics::neighbors::KDTree::_find_knn(const arma::rowvec& pt, std::priority_queue<_DistanceIndex>& kbest, const u_long& k) const {
    _SearchNode sn;
    sn.node = _head;
    sn.box_distances = arma::max(pt - _bounding_box.row(1), _bounding_box.row(0) - pt);
    sn.box_distances.clean(0.0);
    if (_p > 1) sn.box_distances = arma::pow(sn.box_distances, _p);

    if (_p > 0u) sn.distance = arma::sum(sn.box_distances);
    else sn.distance = arma::norm(sn.box_distances, "inf");

    std::queue<_SearchNode> q;
    q.push(sn);
    while (not q.empty()) {
        _SearchNode t = q.front();
        q.pop();

        std::vector<double> distances;
        for (u_long i :  t.node->index) {
            double dist = _distance(pt, _data.row(i));
            if (kbest.size() < k) {
                kbest.push({dist, i});
            } else if (dist < kbest.top().distance) {
                kbest.pop();
                kbest.push({dist, i});
            }
        }

        _Node *closer, *further;
        u_long i = t.node->index.front();
        if (pt(t.node->dim) < _data(i, t.node->dim)) {
            closer = t.node->left;
            further = t.node->right;
        } else {
            closer = t.node->right;
            further = t.node->left;
        }
        
        if (closer != nullptr) {
            sn.node = closer;
            sn.distance = t.distance;
            sn.box_distances = t.box_distances;
            q.push(sn);
        }

        if (further != nullptr) {
            u_long j = t.node->dim;
            sn.node = further;
            sn.box_distances = t.box_distances;
            sn.box_distances(t.node->dim) = std::abs(_data(i, j) - pt(j));
            if (_p > 1) sn.box_distances(j) = std::pow(sn.box_distances(j), _p);

            if (_p >= 1) sn.distance = t.distance - t.box_distances(j) + sn.box_distances(j);
            else sn.distance = std::max(t.distance, sn.box_distances(j));
            
            if (sn.distance < kbest.top().distance) q.push(sn);
        }
    }
}

void numerics::neighbors::KDTree::_query(arma::mat& distances, arma::umat& neighbors, const arma::mat& pts, u_long k, bool return_distances) const {
    if (pts.n_cols != _n_dims) {
        throw std::invalid_argument("number of columns in query pts (=" + std::to_string(pts.n_cols) + ") does not equal the dimension of the KDTree (=" +std::to_string(_n_dims) + ").");
    }
    neighbors.set_size(pts.n_rows, k);
    if (return_distances) distances.set_size(pts.n_rows, k);

    arma::vec ds; ds.set_size(k);
    arma::uvec ns; ns.set_size(k);
    for (u_long i=0; i < pts.n_rows; ++i) {
        std::priority_queue<_DistanceIndex> kbest;

        _find_knn(pts.row(i), kbest, k);
        for (u_long j=0; j < k; ++j) {
            ds(j) = kbest.top().distance;
            ns(j) = kbest.top().index;
            kbest.pop();
        }
        arma::uvec idx = arma::sort_index(ds);
        neighbors.row(i) = ns(idx).as_row();
        if (return_distances) distances.row(i) = ds(idx).as_row();
    }
}

void numerics::neighbors::KDTree::_clear_tree(_Node* node) {
    if (node != nullptr) {
        _clear_tree(node->left);
        _clear_tree(node->right);
        delete node;
    }
}

void numerics::neighbors::KDTree::_copy_tree(_Node* node, _Node* copy_node) {
    if (copy_node != nullptr) {
        if (node == nullptr) node = new _Node();
        node->index = copy_node->index;
        node->dim = copy_node->dim;
        _copy_tree(node->left, copy_node->left);
        _copy_tree(node->right, copy_node->right);
    }
}

void numerics::neighbors::KDTree::fit(const arma::mat& x) {
    _size = x.n_rows;
    _n_dims = x.n_cols;

    _data = x;

    _build_tree();
    
    _bounding_box.set_size(2, _n_dims);
    for (u_long i=0; i < _n_dims; ++i) {
        _bounding_box(0, i) = x.col(i).min();
        _bounding_box(1, i) = x.col(i).max();
    }
}

void numerics::neighbors::KDTree::fit(arma::mat&& x) {
    _size = x.n_rows;
    _n_dims = x.n_cols;

    _data = x;

    _build_tree();
    
    _bounding_box.set_size(2, _n_dims);
    for (u_long i=0; i < _n_dims; ++i) {
        _bounding_box(0, i) = x.col(i).min();
        _bounding_box(1, i) = x.col(i).max();
    }
}

double numerics::neighbors::KDTree::min(u_long dim) const {
    return _bounding_box(0, dim);
}

double numerics::neighbors::KDTree::max(u_long dim) const {
    return _bounding_box(1, dim);
}

void numerics::neighbors::KDTree::query(arma::mat& distances, arma::umat& neighbors, const arma::mat& pts, u_long k) const {
    _query(distances, neighbors, pts, k, true);
}

void numerics::neighbors::KDTree::query(arma::umat& neighbors, const arma::mat& pts, u_long k) const {
    arma::mat dummy;
    _query(dummy, neighbors, pts, k, false);
}
