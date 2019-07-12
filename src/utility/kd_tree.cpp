#include <numerics.hpp>

/* kd_tree(data) : construct kd_tree from entire data set allowing the tree to be maximally balanced. This implementation is designed for applications where insertion and deletion operations are unnecessary. Construction is O(n log n). */
numerics::numerics_private_utility::kd_tree_util::kd_tree::kd_tree(const arma::mat& data) {
    int n_pt = data.n_rows;
    int k = data.n_cols;
    if (std::pow(2,k) > n_pt) {
        std::cerr << "kd_tree warning: the dimension of the data may be too large for the tree to be efficient.\n";
    }

    X = data;
    int d = arma::as_scalar(arma::index_max( arma::var(data), 1));
    first_split = d;
    head = build_tree(X, arma::regspace<arma::uvec>(0,data.n_rows-1), d);
    bounding_box = arma::zeros(2,X.n_cols);
    for (uint i=0; i < X.n_cols; ++i) { // find lower and upper limit of subtree
        bounding_box(0,i) = min(i);
        bounding_box(1,i) = max(i);
    }
}

/* build_tree(data,indices,current_dim) *private* recursively builds tree nodes from data storing only the row-indicies from the original data matrix. */
numerics::numerics_private_utility::kd_tree_util::node* numerics::numerics_private_utility::kd_tree_util::kd_tree::build_tree(const arma::mat& data, const arma::uvec& inds, int d) {
    if (data.n_rows < 1) return nullptr;
    if (data.n_rows==1) {
        node* A = new node;
        A->ind = inds(0);
        return A;
    }
    d = d%(data.n_cols);
    uint m_i = index_median(data.col(d));
    arma::uvec xless = arma::find(data.col(d) < data(m_i,d));
    arma::uvec xgreater = arma::find(data.col(d) >= data(m_i,d) && inds != inds(m_i));
    
    node* A = new node;
    A->ind = inds(m_i);
    A->left_child = build_tree(data.rows(xless), inds(xless), d+1);
    if (A->left_child != nullptr) A->left_child->parent = A;
    A->right_child = build_tree(data.rows(xgreater), inds(xgreater), d+1);
    if (A->right_child != nullptr) A->right_child->parent = A;
    return A;
}

/* find_min(node, dim, current_dim) *private* recursive function for finding the minimum value along the axis dim.  */
double numerics::numerics_private_utility::kd_tree_util::kd_tree::find_min(node* T, uint dim, uint current_dim) {
    if (T == nullptr) return NAN;
    if (current_dim == dim) {
        if (T->left_child == nullptr) return X(T->ind, dim);
        else return find_min(T->left_child, dim, (current_dim+1)%X.n_cols);
    } else {
        double minleft = find_min(T->left_child,dim,(current_dim+1)%X.n_cols);
        double minright = find_min(T->right_child,dim,(current_dim+1)%X.n_cols);
        return std::min<double>({X(T->ind,dim),minleft,minright});
    }
}

/* find_max(node, dim, current_dim) *private* recursive function for finding the maximum value along the axis dim. */
double numerics::numerics_private_utility::kd_tree_util::kd_tree::find_max(node* T, uint dim, uint current_dim) {
    if (T == nullptr) return NAN;
    if (current_dim == dim) {
        if (T->right_child == nullptr) return X(T->ind, dim);
        else return find_max(T->right_child, dim, (current_dim+1)%X.n_cols);
    } else {
        double maxleft = find_max(T->left_child,dim,(current_dim+1)%X.n_cols);
        double maxright = find_max(T->right_child,dim,(current_dim+1)%X.n_cols);
        return std::max<double>({X(T->ind,dim),maxleft,maxright});
    }    
}

/* find_kNN(query_pt, node, current_dim, bounding_box, kbest, k) *private* recursive function for finding the k nearest neighbors to the query_pt updating kbest along the way. */
void numerics::numerics_private_utility::kd_tree_util::kd_tree::find_kNN(const arma::rowvec& pt, node* T, uint current_dim, const arma::mat& bounds, pqueue& kbest, const uint& k) {
    if (T == nullptr) return;
    double dist = arma::norm(X.row(T->ind) - pt);
    if (kbest.size() < k) {
        dist_ind DI;
        DI.dist = dist;
        DI.ind = T->ind;
        kbest.push(DI);
    } else {
        arma::rowvec center = arma::mean(bounds);
        arma::rowvec center_diff = arma::abs(pt - center) - arma::abs(bounds.row(0) - center);
        center_diff(arma::find(center_diff < 0)).zeros();
        double dist_outside = arma::norm(center_diff);

        if (dist_outside > kbest.top().dist) return;
        
        if (dist < kbest.top().dist) {
            dist_ind DI;
            DI.dist = dist;
            DI.ind = T->ind;
            kbest.pop();
            kbest.push(DI);
        }
    }
    arma::mat trimlower = bounds;
    trimlower(0,current_dim) = X(T->ind, current_dim);
    arma::mat trimupper = bounds;
    trimupper(1,current_dim) = X(T->ind,current_dim);
    if (pt(current_dim) > X(T->ind,current_dim)) {
        find_kNN(pt, T->left_child, (current_dim+1)%pt.n_elem, trimlower, kbest, k);
        find_kNN(pt, T->right_child, (current_dim+1)%pt.n_elem, trimupper, kbest, k);
    } else {
        find_kNN(pt, T->left_child, (current_dim+1)%pt.n_elem, trimupper, kbest, k);
        find_kNN(pt, T->right_child, (current_dim+1)%pt.n_elem, trimlower, kbest, k);
    }
}

/* min(dim) : find minimum value along axis dim.
 * --- dim : axis along which to find the minimum */
double numerics::numerics_private_utility::kd_tree_util::kd_tree::min(uint dim) {
    return find_min(head, dim, first_split);
}

/* max(dim) : find maximum value along axis dim. 
 * --- dim : axis along which to find the maximum. */
double numerics::numerics_private_utility::kd_tree_util::kd_tree::max(uint dim) {
    return find_max(head, dim, first_split);
}

/* index_kNN(query_pt, k) : find the index (relative to the original data matrix) of the k nearest neighbors of query_pt. Method is average case O(log k * log n).
 * --- query_pt : new data point to find nearest neighbors of.
 * --- k : number of nearest neighbors. */
arma::uvec numerics::numerics_private_utility::kd_tree_util::kd_tree::index_kNN(const arma::rowvec& pt, uint k) {
    pqueue kbest;
    find_kNN(pt, head, first_split, bounding_box, kbest, k);
    arma::uvec kNN = arma::zeros<arma::uvec>(k);
    for (int i=k-1; i >= 0; --i) {
        kNN(i) = kbest.top().ind;
        kbest.pop();
    }
    return kNN;
}

/* find_kNN(query_pt, k) : return the k nearest neighbors of the query_pt. Method is average case O(log k * log n).
 * --- query_pt : new data point to find the nearest neighbors of.
 * --- k : number of nearest neighbors. */
arma::mat numerics::numerics_private_utility::kd_tree_util::kd_tree::find_kNN(const arma::rowvec& pt, uint k) {
    arma::uvec ind = index_kNN(pt,k);
    return X.rows(ind);
}

arma::mat numerics::numerics_private_utility::kd_tree_util::kd_tree::data() {
    return X;
}

uint numerics::numerics_private_utility::kd_tree_util::kd_tree::size() {
    return X.n_rows;
}

uint numerics::numerics_private_utility::kd_tree_util::kd_tree::dim() {
    return X.n_cols;
}