#ifndef NUMERICS_DATA_SCIENCE_HPP
#define NUMERICS_DATA_SCIENCE_HPP

template<typename eT> class AutoEstimator {
    protected:
    u_long _dim;
    void _check_x(const arma::mat& x) const { // always call at beginning of predict
        if (x.n_cols != _dim) {
            throw std::invalid_argument("dimension mismatch, (query.n_cols = " + std::to_string(x.n_cols) + ") which does not equal the dimension of the fitted data (dim = " + std::to_string(_dim) + ").");
        }
    }

    public:
    virtual void fit(const arma::mat& x) = 0;
    virtual arma::Col<eT> fit_predict(const arma::mat& x) = 0;
    virtual arma::Col<eT> predict(const arma::mat& x) const = 0;
};

template<typename eT> class Estimator {
    protected:
    u_long _dim;
    void _check_x(const arma::mat& x) const { // call for predict
        if (x.n_cols != _dim) {
            throw std::invalid_argument("dimension mismatch, (query.n_cols = " + std::to_string(x.n_cols) + ") which does not equal the dimension of the fitted data (dim = " + std::to_string(_dim) + ").");
        }
    }
    void _check_xy(const arma::mat& x, const arma::Col<eT>& y) const { // call for fit and score
        if (x.n_rows != y.n_rows) {
            throw std::invalid_argument("require number of rows in X (=" + std::to_string(x.n_rows) + ") == number of rows in y (=" + std::to_string(y.n_rows) + ").");
        }
    }
    
    public:
    virtual void fit(const arma::mat& x, const arma::Col<eT>& y) = 0;
    virtual arma::Col<eT> predict(const arma::mat& x) const = 0;
    virtual double score(const arma::mat& x, const arma::Col<eT>& y) const = 0;
};

typedef Estimator<double> Regressor;
typedef Estimator<arma::uword> Classifier;

/* mse_score: returns the mean squared error between a vector of known values and their estimated values.
 * --- y : true values.
 * --- yhat : predicted values. */
inline double mse_score(const arma::vec& y, const arma::vec& yhat) {
    return arma::mean(arma::square(y - yhat));
}

/* r2_score: returns the coefficient of determination between a vector of known values and their estimated values.
 * --- y : true values.
 * --- yhat : predicted values. */
inline double r2_score(const arma::vec& y, const arma::vec& yhat) {
    return 1.0 - mse_score(y, yhat)/arma::var(arma::conv_to<arma::vec>::from(y),1);
}

/* accuracy_score: returns the accuracy score between a vector of known values and predicted values.
 * --- y : true values.
 * --- yhat : predicted values. */
inline double accuracy_score(const arma::uvec& y, const arma::uvec& yhat) {
    return (long double)arma::sum(y == yhat) / y.n_elem;
}

template<typename A> class KFolds1Arr {
    protected:
    u_int _n_folds;
    arma::Mat<A> _X;
    arma::umat _I;
    arma::uvec _range;

    void _check_index(u_long i) {
        if (i >= _n_folds) {
            throw std::out_of_range("i (=" + std::to_string(i) + ") out of range of KFolds (size=" + std::to_string(_n_folds) + ").");
        }
    }

    public:
    const arma::Mat<A>& X;

    explicit KFolds1Arr(int n_folds=2) : X(_X) {
        if (n_folds < 2) {
            throw std::invalid_argument("require number of folds (=" + std::to_string(n_folds) + ") >= 2.");
        }
        _n_folds = n_folds;
    }

    void fit(arma::Mat<A>&& xx) {
        _X = xx;

        _I = arma::reshape(arma::randperm(_X.n_rows), _X.n_rows/_n_folds, _n_folds);
        _range = arma::regspace<arma::uvec>(0, _n_folds-1);
    }

    void fit(const arma::Mat<A>& xx) {
        _X = xx;

        _I = arma::reshape(arma::randperm(_X.n_rows), _X.n_rows/_n_folds, _n_folds);
        _range = arma::regspace<arma::uvec>(0, _n_folds-1);
    }

    arma::Mat<A> train(u_long i) {
        _check_index(i);
        arma::umat ii = _I.cols( arma::find(_range != i) );
        return X.rows(ii.as_col());
    }

    arma::Mat<A> test(u_long i) {
        _check_index(i);
        return X.rows(_I.col(i));
    }
};

template<typename A, typename B> class KFolds2Arr {
    protected:
    u_int _n_folds;
    arma::Mat<A> _X;
    arma::Mat<B> _y;
    arma::umat _I;
    arma::uvec _range;
    bool _move;

    void _check_index(u_long i) const {
        if (i >= _n_folds) {
            throw std::out_of_range("i (=" + std::to_string(i) + ") out of range of KFolds (size=" + std::to_string(_n_folds) + ").");
        }
    }

    public:
    const arma::Mat<A>& X;
    const arma::Mat<B>& y;

    explicit KFolds2Arr(int n_folds=2) : X(_X), y(_y) {
        if (n_folds < 2) {
            throw std::invalid_argument("require number of folds (=" + std::to_string(n_folds) + ") >= 2.");
        }
        _n_folds = n_folds;
    }
    
    void fit(const arma::Mat<A>& xx, const arma::Mat<B>& yy) {
        if (xx.n_rows != yy.n_rows) {
            throw std::invalid_argument("require number of rows in X (=" + std::to_string(xx.n_rows) + ") == number of rows in y (=" + std::to_string(yy.n_rows) + ").");
        }
        _X = xx;
        _y = yy;
        _I = arma::reshape(arma::randperm(_X.n_rows), _X.n_rows/_n_folds, _n_folds);
        _range = arma::regspace<arma::uvec>(0, _n_folds-1);
    }

    void fit(arma::Mat<A>&& xx, arma::Mat<B>&& yy) {
        if (xx.n_rows != yy.n_rows) {
            throw std::invalid_argument("require number of rows in X (=" + std::to_string(xx.n_rows) + ") == number of rows in y (=" + std::to_string(yy.n_rows) + ").");
        }
        _X = xx;
        _y = yy;
        _I = arma::reshape(arma::randperm(_X.n_rows), _X.n_rows/_n_folds, _n_folds);
        _range = arma::regspace<arma::uvec>(0, _n_folds-1);
    }
    
    arma::Mat<A> trainX(u_long i) const {
        _check_index(i);
        arma::umat ii = _I.cols( arma::find(_range != i) );
        return X.rows(ii.as_col());
    }
    
    arma::Mat<B> trainY(u_long i) const {
        _check_index(i);
        arma::umat ii = _I.cols( arma::find(_range != i) );
        return y.rows(ii.as_col());
    }
    
    arma::Mat<A> testX(u_long i) const {
        _check_index(i);
        return X.rows( _I.col(i) );
    }
    
    arma::Mat<B> testY(u_long i) const {
        _check_index(i);
        return y.rows( _I.col(i) );
    }
};

typedef KFolds2Arr<double,double> KFolds;

template<class T> class LabelEncoder {
    protected:
    std::map<T,arma::uword> _classes;
    std::map<arma::uword,T> _iclasses;
    arma::uword _n_classes;

    public:
    LabelEncoder() {};
    void fit(const std::vector<T>& y) {
        _n_classes=0;
        _classes.clear();
        _iclasses.clear();
        for (const T& t : y) {
            if (_classes.count(t) == 0) {
                _classes.insert({t,_n_classes});
                _iclasses.insert({_n_classes,t});
                _n_classes++;
            }
        }
    }

    arma::uvec encode(const std::vector<T>& y) const {
        arma::uvec yh; yh.set_size(y.size());
        for (u_long i=0; i < y.size(); ++i) {
            yh(i) = _classes.at(y.at(i));
        }
        return yh;
    }

    std::vector<T> decode(const arma::uvec& y) const {
        std::vector<T> yh;
        for (u_long i=0; i < y.n_elem; ++i) {
            yh.push_back(_iclasses.at(y(i)));
        }
        return yh;
    }
};

class OneHotEncoder {
    protected:
    arma::uvec _classes;
    bool _pm;

    public:
    /* converts vector of classes to one-hot-encoding. A value of 1 is always the class indicator. By default the other values are 0, but specifying plus_minus=true sets the other values to -1. */
    explicit OneHotEncoder(bool plus_minus=false) {
        _pm = plus_minus;
    }
    void fit(const arma::uvec& x) {
        _classes = arma::unique(x);
    }
    arma::mat encode(const arma::uvec& x) const {
        arma::mat onehot;
        if (_pm) onehot = -arma::ones(x.n_elem, _classes.n_elem);
        else onehot = arma::zeros(x.n_elem, _classes.n_elem);
        for (u_long i=0; i < x.n_elem; ++i) {
            bool in_range = false;
            for (u_long j=0; j < _classes.n_elem; ++j) {
                if (x(i) == _classes(j)) {
                    onehot(i,j) = 1;
                    in_range = true;
                    break;
                }
            }
            if (not in_range) {
                std::string er = "detected class (=" + std::to_string(x(i)) + ") not in the set of fitted classes {";
                for (arma::uword c : _classes) er += std::to_string(c) + ",";
                er += "}";
                throw std::range_error(er);
            }
        }
        return onehot;
    }
    
    /* decode one hot encoded values by their class labels. The decoder does not expect a valid encoding (0/1 or -1/1), the inference is based on maximum values. */
    arma::uvec decode(const arma::mat& x) const {
        if (x.n_cols != _classes.n_elem) {
            throw std::invalid_argument("dimension mismatch, (query.n_cols = " + std::to_string(x.n_cols) + ") which does not equal the dimension of the fitted data (dim = " + std::to_string(_classes.n_elem) + ").");
        }
        return _classes(arma::index_max(x, 1));
    }
};

class BinData {
    private:
    u_long _n;
    double _bin_width;
    arma::vec _bins;
    arma::vec _counts;

    void _set_bins(u_long n_obs);

    public:
    const u_long& n_bins;
    const double& bin_width;
    const arma::vec& bins;
    const arma::vec& counts;

    explicit BinData(long bins=0) : n_bins(_n), bin_width(_bin_width), bins(_bins), counts(_counts) {
        if (bins < 0) {
            throw std::invalid_argument("require bins (=" + std::to_string(bins) + ") >= 0, (0 indicates auto-select).");
        }
        _n = bins;
    };
    
    /* to_bins(x) : place data into regularly spaced 'bins' with non-integer 'counts' corresponding to a linear distance weights.
        * e.g. if bin1 = 0 and bin2 = 1 and x = 0.2 then bin1 would have a count of 0.8 and bin2 would have a count of 0.2.
        * --- x : data to place into bins. */
    void fit(const arma::mat& x);

    /* to_bins(x, y) : place data into regularly spaced 'bins' with non-integer 'counts' corresponding to a linear distance weights of the dependent variable y.
        * e.g. if bin1 = 0 and bin2 = 1 and x = 0.2 then bin1 would have a count of 0.8 and bin2 would have a count of 0.2.
        * --- x : data to place into bins.
        * --- y : dependant variable */
    void fit(const arma::mat& x, const arma::vec& y);
};

namespace neighbors {
    struct _Node {
        public:
        _Node *left;
        _Node *right;

        std::vector<u_long> index;
        u_long dim;

        _Node() {
            left = nullptr;
            right = nullptr;
        }
    };

    struct _DistanceIndex {
        public:
        double distance;
        u_long index;
        _DistanceIndex(double D, u_long I) : distance(D), index(I) {}
        bool operator<(const _DistanceIndex& a) const {
            return distance < a.distance;
        }
    };

    struct _SearchNode {
        public:
        double distance;
        _Node* node;
        arma::rowvec box_distances;
    };

    class KDTree {
        protected:
        u_long _size;
        u_long _n_dims;
        u_long _leaf_size;
        u_long _p;
        
        arma::mat _data;
        arma::mat _bounding_box;
        _Node *_head;

        void _valid_leaf_size(int leaf_size);
        void _valid_p_norm(int pnorm);
        double _distance(const arma::rowvec& a, const arma::rowvec& b) const;
        u_long _index_median(const std::vector<u_long>& rows, u_long col);
        void _find_rel(std::vector<u_long>& less, std::vector<u_long>& greater, const std::vector<u_long>& rows, u_long col, u_long pivot);
        void _build_tree();
        void _copy_tree(_Node* node, _Node* copy_node);
        void _clear_tree(_Node* node);
        u_long _next_dim(u_long d);
        void _find_knn(const arma::rowvec& pt, std::priority_queue<_DistanceIndex>& kbest, const u_long& k) const;
        void _query(arma::mat& distances, arma::umat& neighbors, const arma::mat& pts, u_long k, bool return_distances) const;

        public:
        const arma::mat& data; // read only view of the data
        const u_long& p_norm;
        const u_long& leaf_size;

        /* KDTree : implements kd-tree data structure for O(log n) nearest neighbor searches.
            * --- p_norm : distance type, >=0 or "inf", (0 indicates "inf").
            * --- leaf_size : number of points to store in the leaf nodes. Storing the data this way takes advantage of the fast brute force search for small data-sets and the fast tree search for large data. */
        explicit KDTree(int pnorm=2, int leafsize=30) : data(_data), p_norm(_p), leaf_size(_leaf_size) {
            _valid_leaf_size(leafsize);
            _leaf_size = leafsize;
            _valid_p_norm(pnorm);
            _p = pnorm;
            _head = nullptr;
        }

        /* KDTree : implements kd-tree data structure for O(log n) nearest neighbor searches.
            * --- p_norm : distance type, >=0 or "inf", (0 indicates "inf").
            * --- leaf_size : number of points to store in the leaf nodes. Storing the data this way takes advantage of the fast brute force search for small data-sets and the fast tree search for large data. */
        explicit KDTree(const std::string& pnorm, int leafsize=30) : data(_data), p_norm(_p), leaf_size(_leaf_size) {
            _valid_leaf_size(leafsize);
            _leaf_size = leafsize;
            
            std::string p;
            for (char c : pnorm) p += std::tolower(c);
            if (p == "inf") _p = 0;
            else {
                throw std::invalid_argument("require p_norm (=" + std::to_string(p_norm) + ") >= 0 or \"inf\" (0 indicates \"inf\").");
            }
            _head = nullptr;
        }

        /* Copy constructor for KDTree. */
        KDTree(const KDTree& T) : data(_data), p_norm(_p), leaf_size(_leaf_size) {
            _size = T._size;
            _n_dims = T._n_dims;
            _leaf_size = T._leaf_size;
            _p = T._p;
            _data = T._data;
            _bounding_box = T._bounding_box;
            _clear_tree(_head); // safely free the memory before copy
            _copy_tree(_head, T._head);
        }

        ~KDTree() {
            _clear_tree(_head);
        }

        /* fit : represents data-set as a tree structure. */
        void fit(const arma::mat& x);
        void fit(arma::mat&& x);
        
        /* min : find minimum along a specified dimension. */
        double min(u_long dim) const;

        /* max : find maximum along a specified dimension. */
        double max(u_long dim) const;

        /* query : find nearest-neighbors for all query points. Returns distances and neighbors indices.
            * --- distances : stores distances here. Size is set to = (pts.n_rows, k), so distances(i,j) = |pts.row(i) - neighbor(j)|.
            * --- neighbors : stores neighbors indices here. Size is set to (pts.n_rows, k), so neighbors(i,j) = j^th closest neighbor to pts.row(i).
            * --- pts : array of points where each row is a point. The number of columns must be equal to the number of columns in the fitted data.
            * --- k : number of neighbors to find. */
        void query(arma::mat& distances, arma::umat& neighbors, const arma::mat& pts, u_long k) const;

        /* query : find nearest-neighbors for all query points. Returns neighbors indices.
            * --- neighbors : stores neighbors indices here. Size is set to (pts.n_rows, k), so neighbors(i,j) = j^th closest neighbor to pts.row(i).
            * --- pts : array of points where each row is a point. The number of columns must be equal to the number of columns in the fitted data.
            * --- k : number of neighbors to find. */
        void query(arma::umat& neighbors, const arma::mat& pts, u_long k) const;
    };
}

template<typename eT> class KNeighborsEstimator : public Estimator<eT> {
    protected:
    arma::Col<eT> _y;
    neighbors::KDTree _data;
    u_int _k;
    arma::uvec _ks;
    arma::vec _scores;
    bool _weighted;

    u_long _dim;
    void _check_x(const arma::mat& x) const{ // call for predict
        if (x.n_cols != _dim) {
            throw std::invalid_argument("dimension mismatch, (query.n_cols = " + std::to_string(x.n_cols) + ") which does not equal the dimension of the fitted data (dim = " + std::to_string(_dim) + ").");
        }
    }
    void _check_xy(const arma::mat& x, const arma::Col<eT>& y) const { // call for fit and score
        if (x.n_rows != y.n_rows) {
            throw std::invalid_argument("require number of rows in X (=" + std::to_string(x.n_rows) + ") == number of rows in y (=" + std::to_string(y.n_rows) + ").");
        }
    }
    void _check_k(int k) {
        if (k <= 0) {
            throw std::invalid_argument("require k (=" + std::to_string(k) + ") >= 1.");
        }
    }

    void _fit(const arma::mat& xx, const arma::Col<eT>& yy) {
        _check_xy(xx, yy);
        _dim = xx.n_cols;
        if (not _ks.is_empty()) {
            KFolds2Arr<double,eT> split(3);
            split.fit(xx, yy);

            _scores = arma::zeros(_ks.n_elem);
            for (u_long i=0; i < _ks.n_elem; ++i) {
                _k = _ks(i);
                for (short j=0; j < 3; ++j) {
                    _data.fit(split.trainX(j));
                    _y = split.trainY(j);
                    _scores(i) += score(split.testX(j), split.testY(j));
                }
            }
            _scores /= 3;
            int best_score = _scores.index_max();
            _k = _ks(best_score);
        }
    }

    virtual eT _vote(const arma::umat& idx, const arma::mat& distances) const = 0;

    virtual eT _vote(const arma::umat& idx) const = 0;

    public:
    const u_int& k;
    const arma::uvec& ks;
    const arma::vec& scores;
    const neighbors::KDTree& X;
    const arma::Col<eT>& y;

    explicit KNeighborsEstimator(int K, int p_norm, bool use_distance_weights, int leaf_size) : _data(p_norm, leaf_size), _weighted(use_distance_weights), k(_k), ks(_ks), scores(_scores), X(_data), y(_y) {
        _check_k(K);
        _k = K;
    }

    explicit KNeighborsEstimator(int K, const std::string& p_norm, bool use_distance_weights,  int leaf_size) : _data(p_norm, leaf_size), _weighted(use_distance_weights), k(_k), ks(_ks), scores(_scores), X(_data), y(_y) {
        _check_k(K);
        _k = K;
    }

    explicit KNeighborsEstimator(const arma::uvec& Ks, int p_norm, bool use_distance_weights, int leaf_size) : _data(p_norm, leaf_size), _weighted(use_distance_weights), k(_k), ks(_ks), scores(_scores), X(_data), y(_y) {
        for (arma::uword K : Ks) _check_k(K);
        _ks = Ks;
    }

    explicit KNeighborsEstimator(const arma::uvec& Ks, const std::string& p_norm, bool use_distance_weights, int leaf_size) : _data(p_norm, leaf_size), _weighted(use_distance_weights), k(_k), ks(_ks), scores(_scores), X(_data), y(_y) {
        for (arma::uword K : Ks) _check_k(K);
        _ks = Ks;
    }

    void fit(const arma::mat& xx, const arma::Col<eT>& yy) override {
        _fit(xx,yy);
        _data.fit(xx);
        _y = yy;
    }

    void fit(arma::mat&& xx, arma::Col<eT>&& yy) {
        _fit(xx,yy);
        _data.fit(xx);
        _y = yy;
    }

    arma::Col<eT> predict(const arma::mat& xx) const {
        _check_x(xx);
        arma::Col<eT> yh; yh.set_size(xx.n_rows);
        arma::umat indices;
        if (_weighted) {
            arma::mat distances;
            _data.query(distances, indices, xx, _k);
            
            distances.each_row(
                [](arma::rowvec& u) -> void {
                    u += 1.0e-8; // prevent zero division.
                    u = 1 / u;
                    u /= arma::sum(u);
                }
            );

            for (u_long i=0; i < xx.n_rows; ++i) {
                yh(i) = _vote(indices.row(i), distances.row(i));
            }
        } else {
            _data.query(indices, xx, _k);

            for (u_long i=0; i < xx.n_rows; ++i) {
                yh(i) = _vote(indices.row(i));
            }
        }
        return yh;
    };

    virtual double score(const arma::mat& xx, const arma::Col<eT>& yy) const = 0;
};

class KNeighborsClassifier : public KNeighborsEstimator<arma::uword> {
    protected:
    arma::uvec _classes;
    arma::uword _vote(const arma::umat& idx, const arma::mat& distances) const override {
        arma::uvec yyh = _y(idx);
        arma::uvec c = arma::unique(yyh);
        arma::vec votes = arma::zeros(c.n_elem);
        for (u_int j=0; j < c.n_elem; ++j) {
            arma::uvec ii = arma::find(yyh == c(j));
            votes(j) += arma::sum(distances(ii));
        }
        u_int top = votes.index_max();
        return c(top);
    }

    arma::uword _vote(const arma::umat& idx) const override {
        arma::uvec yyh = _y(idx);
        arma::uvec c = arma::unique(yyh);
        arma::vec votes = arma::zeros(c.n_elem);
        for (u_int j=0; j < c.n_elem; ++j) {
            arma::uvec ii = arma::find(yyh == c(j));
            votes(j) += ii.n_elem;
        }
        u_int top = votes.index_max();
        return c(top);
    }

    public:
    /* initializes k-neighbors classifier, by specifying k or a vector of k values to test by cross-validation.
        * --- K : number of nearest neighbors to use for estimation.
        * --- p_norm : type of distance, either int >= 0 or "inf" (0 indicates "inf")
        * --- use_distance_weights : weigh estimate by the distance of neighbors.
        * --- leaf_size : leaf_size for KDTree object for nearest neighbor queries.
        * --- move_data : whether to move data into KDTree or copy. */
    explicit KNeighborsClassifier(int K, int p_norm=2, bool use_distance_weights=false, int leaf_size=30) : KNeighborsEstimator<arma::uword>(K,p_norm,use_distance_weights,leaf_size) {}
    
    /* initializes k-neighbors classifier, by specifying k or a vector of k values to test by cross-validation.
        * --- K : number of nearest neighbors to use for estimation.
        * --- p_norm : type of distance, either int >= 0 or "inf" (0 indicates "inf")
        * --- use_distance_weights : weigh estimate by the distance of neighbors.
        * --- leaf_size : leaf_size for KDTree object for nearest neighbor queries.
        * --- move_data : whether to move data into KDTree or copy. */
    explicit KNeighborsClassifier(int K, const std::string& p_norm, bool use_distance_weights=false,  int leaf_size=30) : KNeighborsEstimator<arma::uword>(K,p_norm,use_distance_weights,leaf_size) {}
    
    /* initializes k-neighbors classifier, by specifying k or a vector of k values to test by cross-validation.
        * --- Ks : list of nearest neighbors to test (using cross-validation).
        * --- p_norm : type of distance, either int >= 1 or "inf"
        * --- use_distance_weights : weigh estimate by the distance of neighbors.
        * --- leaf_size : leaf_size for KDTree object for nearest neighbor queries.
        * --- move_data : whether to move data into KDTree or copy. */
    explicit KNeighborsClassifier(const arma::uvec& Ks, int p_norm=2, bool use_distance_weights=false, int leaf_size=30) : KNeighborsEstimator<arma::uword>(Ks,p_norm,use_distance_weights,leaf_size) {}
    
    /* initializes k-neighbors classifier, by specifying k or a vector of k values to test by cross-validation.
        * --- Ks : list of nearest neighbors to test (using cross-validation).
        * --- p_norm : type of distance, either int >= 1 or "inf"
        * --- use_distance_weights : weigh estimate by the distance of neighbors.
        * --- leaf_size : leaf_size for KDTree object for nearest neighbor queries.
        * --- move_data : whether to move data into KDTree or copy. */
    explicit KNeighborsClassifier(const arma::uvec& Ks, const std::string& p_norm, bool use_distance_weights=false, int leaf_size=30) : KNeighborsEstimator<arma::uword>(Ks,p_norm,use_distance_weights,leaf_size) {}
    
    /* returns the mean accuracy between yy and predict(xx) */
    double score(const arma::mat& xx, const arma::uvec& yy) const override {
        _check_xy(xx, yy);
        return accuracy_score(yy, predict(xx));
    };

    arma::mat predict_proba(const arma::mat& xx) const {
        _check_x(xx);
        arma::uvec c = arma::unique(_y);
        arma::mat yh; yh.set_size(xx.n_rows, c.n_elem);
        arma::umat indices;
        if (_weighted) {
            arma::mat distances;
            _data.query(distances, indices, xx, _k);
            
            distances.each_row(
                [](arma::rowvec& u) -> void {
                    u += 1.0e-8; // prevent zero division.
                    u = 1 / u;
                    u /= arma::sum(u);
                }
            );

            for (u_long i=0; i < xx.n_rows; ++i) {
                arma::uvec yyh = _y(indices.row(i));
                arma::rowvec d = distances.row(i);
                for (u_int j=0; j < c.n_elem; ++j) {
                    arma::uvec ii = arma::find(yyh == c(j));
                    yh(i,j) += arma::sum(d(ii));
                }
            }
        } else {
            _data.query(indices, xx, _k);

            for (u_long i=0; i < xx.n_rows; ++i) {
                arma::uvec yyh = _y(indices.row(i));
                for (u_int j=0; j < c.n_elem; ++j) {
                    arma::uvec ii = arma::find(yyh == c(j));
                    yh(i,j) += ii.n_elem;
                }
            }
        }
        yh.each_row([&](arma::rowvec& u) -> void {u /= arma::accu(u);});
        return yh;
    }
};

class KNeighborsRegressor : public KNeighborsEstimator<double> {
    protected:
    double _vote(const arma::umat& idx, const arma::mat& distances) const override {
        return arma::dot(_y(idx), distances);
    }

    double _vote(const arma::umat& idx) const override {
        return arma::mean(_y(idx));
    }

    public:
    /* initializes k-neighbors regressor, by specifying k or a vector of k values to test by cross-validation.
        * --- K : number of nearest neighbors to use for estimation.
        * --- p_norm : type of distance, either int >= 0 or "inf" (0 indicates "inf")
        * --- use_distance_weights : weigh estimate by the distance of neighbors.
        * --- leaf_size : leaf_size for KDTree object for nearest neighbor queries.
        * --- move_data : whether to move data into KDTree or copy. */
    explicit KNeighborsRegressor(int K, int p_norm=2, bool use_distance_weights=false, int leaf_size=30) : KNeighborsEstimator<double>(K,p_norm,use_distance_weights,leaf_size) {}
    
    /* initializes k-neighbors regressor, by specifying k or a vector of k values to test by cross-validation.
        * --- K : number of nearest neighbors to use for estimation.
        * --- p_norm : type of distance, either int >= 0 or "inf" (0 indicates "inf")
        * --- use_distance_weights : weigh estimate by the distance of neighbors.
        * --- leaf_size : leaf_size for KDTree object for nearest neighbor queries.
        * --- move_data : whether to move data into KDTree or copy. */
    explicit KNeighborsRegressor(int K, const std::string& p_norm, bool use_distance_weights=false,  int leaf_size=30) : KNeighborsEstimator<double>(K,p_norm,use_distance_weights,leaf_size) {}
    
    /* initializes k-neighbors regressor, by specifying k or a vector of k values to test by cross-validation.
        * --- Ks : list of nearest neighbors to test (using cross-validation).
        * --- p_norm : type of distance, either int >= 1 or "inf"
        * --- use_distance_weights : weigh estimate by the distance of neighbors.
        * --- leaf_size : leaf_size for KDTree object for nearest neighbor queries.
        * --- move_data : whether to move data into KDTree or copy. */
    explicit KNeighborsRegressor(const arma::uvec& Ks, int p_norm=2, bool use_distance_weights=false, int leaf_size=30) : KNeighborsEstimator<double>(Ks,p_norm,use_distance_weights,leaf_size) {}
    
    /* initializes k-neighbors regressor, by specifying k or a vector of k values to test by cross-validation.
        * --- Ks : list of nearest neighbors to test (using cross-validation).
        * --- p_norm : type of distance, either int >= 1 or "inf"
        * --- use_distance_weights : weigh estimate by the distance of neighbors.
        * --- leaf_size : leaf_size for KDTree object for nearest neighbor queries.
        * --- move_data : whether to move data into KDTree or copy. */
    explicit KNeighborsRegressor(const arma::uvec& Ks, const std::string& p_norm, bool use_distance_weights=false, int leaf_size=30) : KNeighborsEstimator<double>(Ks,p_norm,use_distance_weights,leaf_size) {}
    
    /* returns the R^2 score between yy and predict(xx) */
    double score(const arma::mat& xx, const arma::vec& yy) const override {
        _check_xy(xx, yy);
        return r2_score(yy, predict(xx));
    }
};

class KMeans : public AutoEstimator<arma::uword> {
    protected:
    arma::mat _clusters, _intra_dist, _nearest_point;
    arma::uvec _nearest_idx;
    uint _k, _max_iter, _p;
    double _tol;
    void _update_labels(arma::uvec& labels, arma::vec& nearest_dist, double& sum_p, const arma::mat& data, uint m);
    void _update_labels(arma::uvec& labels, const arma::mat& data) const;
    void _update_labels_nearest(arma::uvec& labels, const arma::mat& data);
    void _init_clusters(arma::uvec& labels, const arma::mat& data);
    void _update_intra_dist();
    double _d(const arma::rowvec& a, const arma::rowvec& b) const;
    void _check_k(int k);
    void _check_maxiter(long m);
    void _check_tol(double tol);

    public:
    const arma::mat& clusters;
    const arma::mat& cluster_distances;
    const arma::mat& points_nearest_centers;
    const arma::uvec& index_nearest_centers;
    
    /* initializes KMeans object.
        * --- k : number of clusters to compute.
        * --- p_norm : norm for computing differences, p_norm >= 0 or "inf" (0 indicates "inf").
        * --- tol : (>= 0) tolerance for stopping criteria, this is the maximum difference between iterations, i.e. max_{i,j} |c_{i,j}^n - c_{i,j}^{n+1}|, if tol == 0, then update checks will not be computed and instead max_iter will be used as the only stopping criteria.
        * --- max_iter : maximum number of iterations before premature stopping. */
    explicit KMeans(int k, const std::string& pnorm, double tol=1e-2, long max_iter=100) : clusters(_clusters), cluster_distances(_intra_dist), points_nearest_centers(_nearest_point), index_nearest_centers(_nearest_idx) {
        _check_k(k);
        std::string p;
        for (char c : pnorm) p += std::tolower(c);
        if (p == "inf") _p = 0;
        else {
            throw std::invalid_argument("require p_norm (=" + pnorm + ") >= 0 or \"inf\" (0 indicates \"inf\").");
        }
        _check_maxiter(max_iter);
        _max_iter = max_iter;
        _check_tol(tol);
        _tol = tol;
    }
    
    /* initializes KMeans object.
        * --- k : number of clusters to compute.
        * --- p_norm : norm for computing differences, p_norm >= 0 or "inf" (0 indicates "inf").
        * --- tol : (>= 0) tolerance for stopping criteria, this is the maximum difference between iterations, i.e. max_{i,j} |c_{i,j}^n - c_{i,j}^{n+1}|, if tol == 0, then update checks will not be computed and instead max_iter will be used as the only stopping criteria.
        * --- max_iter : maximum number of iterations before premature stopping. */
    explicit KMeans(int k, int p_norm=2, double tol=1e-2, long max_iter=100) : clusters(_clusters), cluster_distances(_intra_dist), points_nearest_centers(_nearest_point), index_nearest_centers(_nearest_idx) {
        _check_k(k);
        _k = k;
        if (p_norm < 0) {
            throw std::invalid_argument("require p_norm (=" + std::to_string(p_norm) + ") >=0 or \"inf\" (0 indicates \"inf\")");
        }
        _p = p_norm;
        _check_maxiter(max_iter);
        _max_iter = max_iter;
        _check_tol(tol);
        _tol = tol;
    }

    /* fits KMeans object and returns cluster labels for data.
        * --- data : matrix of data (num observations, num features). */
    virtual arma::uvec fit_predict(const arma::mat& data) override;

    /* fit KMeans object.
        * --- data : matrix of data (num observations, num features) */
    virtual void fit(const arma::mat& data) override;

    /* predict(nearest_dist, data) : predict labels for new data and compute the distance between each data point and the nearest cluster.
        * --- nearest_dist : the i^th element is the distance between the i^th observation and the nearest cluster center. */
    arma::uvec predict(const arma::mat& data) const override;
};

class KMeansSGD : public KMeans {
    protected:
    u_long _batch_size;
    void _check_batch(long b);
    void _update_batch_labels(arma::uvec& labels, const arma::mat& data, const arma::uvec& p, uint i, uint f);
    void _sgd_steps(arma::uvec& labels, const arma::mat& data);

    public:
    /* initializes KMeansSGD object.
        * --- k : number of clusters to compute.
        * --- p_norm : norm for computing differences, p_norm >= 0 or "inf" (0 indicates "inf").
        * --- batch_size : 
        * --- tol : (>= 0) tolerance for stopping criteria, this is the maximum difference between iterations, i.e. max_{i,j} |c_{i,j}^n - c_{i,j}^{n+1}|, if tol == 0, then update checks will not be computed and instead max_iter will be used as the only stopping criteria.
        * --- max_iter : maximum number of iterations before premature stopping. */
    explicit KMeansSGD(int k, int p_norm=2, int batch_size=100, double tol=1e-2, long max_iter=100) : KMeans(k, p_norm, tol, max_iter) {
        _check_batch(batch_size);
        _batch_size = batch_size;
    }

    explicit KMeansSGD(int k, const std::string& p_norm, int batch_size=100, double tol=1e-2, long max_iter=100) : KMeans(k, p_norm, tol, max_iter) {
        _check_batch(batch_size);
        _batch_size = batch_size;
    }

    /* fit the data using batch gradient descent.
        * --- data : training data set. */
    void fit(const arma::mat& data) override;

    /* fit the data using batch gradient descent and return prediction.
        * --- data : training data set. */
    arma::uvec fit_predict(const arma::mat& data) override;
};

class PolyFeatures {
    protected:
    u_long _dim;
    std::vector<std::vector<u_int>> _monomials;
    u_int _deg;
    bool _intercept;
    arma::vec _scale;

    void _check_x(const arma::mat& x) const { // call for predict
        if (x.n_cols != _dim) {
            throw std::invalid_argument("dimension mismatch, (query.n_cols = " + std::to_string(x.n_cols) + ") which does not equal the dimension of the fitted data (dim = " + std::to_string(_dim) + ").");
        }
    }

    public:
    explicit PolyFeatures(int degree, bool include_intercept=false) {
        _intercept = include_intercept;
        if (degree < 1) {
            throw std::invalid_argument("require degree (=" + std::to_string(degree) + ") >= 1");
        }
        _deg = degree;
    }

    void fit(const arma::mat& x);
    arma::mat fit_predict(const arma::mat& x);

    arma::mat predict(const arma::mat& x) const;
};

/* Computes cubic kernel Graham matrix for data x. */
arma::mat cubic_kernel(const arma::mat& x);

/* Computes cubic kernel values for x2 in terms of kernels centered at rows of x1 */
arma::mat cubic_kernel(const arma::mat& x1, const arma::mat& x2);

namespace bw { // a selection of bandwidth estimation methods
    inline void check_kernel(const std::string& kernel) {
        std::vector<std::string> Kernels = {"gaussian","square","triangle","parabolic"};
        std::string er = "kernel (=" + kernel + ") must be one of {";
        for (const std::string& k : Kernels) {
            if (k == kernel) return;
            er += k + ",";
        }
        er += "}";
        throw std::invalid_argument(er);
    }
    inline void check_estimator(const std::string& method) {
        std::vector<std::string> BandwidthEstimator = {"rule_of_thumb","min_sd_iqr","plug_in","grid_cv"};
        std::string er = "bandwidth estimator method (=" + method + ") must be one of {";
        for (const std::string& m : BandwidthEstimator) {
            if (m == method) return;
            er += m + ",";
        }
        er += "}";
        throw std::invalid_argument(er);
    }
    arma::vec eval_kernel(const arma::vec& x, const std::string& K="gaussian");
    double dpi(const arma::vec& x, double s=0, const std::string& K="gaussian");
    double dpi_binned(const BinData& bins, double s=0, const std::string& K="gaussian");
    double rot1(int n, double s);
    double rot2(const arma::vec& x, double s = 0);
    double grid_mse(const arma::vec& x, const std::string& K, double s=0, int grid_size=20, bool binning=false);
    double grid_mse(const arma::vec& x, const arma::vec& y, const std::string& K, double s=0, int grid_size=20, bool binning=false);
};

class KernelEstimator {
    protected:
    arma::mat _X;
    BinData _bins;
    double _bdw;
    bool _binning;
    std::string _kern;

    public:
    const arma::mat& X;
    const BinData& bin_data;
    const double& bandwidth;

    explicit KernelEstimator(const std::string& kernel, bool binning, long n_bins) : _bins(n_bins), bin_data(_bins), X(_X), bandwidth(_bdw) {
        bw::check_kernel(kernel);
        _kern = kernel;
        _binning = binning;
        _bdw = -1;
    }

    explicit KernelEstimator(double bdw, const std::string& kernel, bool binning, long n_bins) : _bins(n_bins), bin_data(_bins), X(_X), bandwidth(_bdw) {
        if (bdw <= 0) {
            throw std::invalid_argument("require bdw (=" + std::to_string(bdw) + ") > 0.0");
        }
        _bdw = bdw;
        bw::check_kernel(kernel);
        _kern = kernel;
        _binning = binning;
    }
};

class KernelSmooth : public KernelEstimator, public Regressor {
    protected:
    arma::mat _y;

    public:
    const arma::mat& y;

    /* initializes kernel smoothing object by specifying the kernel, and selecting the bandwidth by cross-validation
     * --- kernel : choice of kernel. options include : gaussian, square, triangle, parabolic
     * --- binning : whether to bin data or not.
     * --- n_bins : number of bins for BinData, ignored if binning == false. */
    explicit KernelSmooth(const std::string& kernel="gaussian", bool binning=true, long n_bins=30) : KernelEstimator(kernel,binning,n_bins), y(_y) {}
    
    /* initializes kernel smoothing object by specifying the kernel and bandwidth
     * --- bdw : bandwidth for kernel estimate.
     * --- kernel : choice of kernel. options include : gaussian, square, triangle, parabolic
     * --- binning : whether to bin data or not.
     * --- n_bins : number of bins for BinData, ignored if binning == false. */
    explicit KernelSmooth(double bdw, const std::string& kernel="gaussian", bool binning=true, long n_bins=30) : KernelEstimator(bdw, kernel, binning, n_bins), y(_y) {}

    /* fits object by storing data, and selecting bandwidth by cross-validation if not specified on construction of the object.
     * --- X : data vector of independent variable
     * --- Y : data vector of dependent variable */
    void fit(const arma::mat& X, const arma::vec& y) override;

    arma::vec predict(const arma::mat& X) const override;

    /* computes R^2 score. */
    double score(const arma::mat& x, const arma::vec& y) const override;
};

class KDE : public KernelEstimator, public AutoEstimator<double> {
    protected:
    std::string _bandwidth_estimator;

    public:
    /* initializes KDE object by specifying a kernel, and a bandwidth selection algorithm.
     * --- kernel : kernel to use. options are: gaussian, square, triangle, parabolic
     * --- bandwidth_estimator : method for estimating the kernel bandwidth. options are: rule_of_thumb, min_sd_iqr, plug_in, grid_cv
     * --- binning : whether to bin the data or not.
     * --- n_bins : number of bins to use for BinData, if binning == false, n_bins is ignored. */
    explicit KDE(const std::string& kernel="gaussian", const std::string& bandwidth_estimator="min_sd_iqr", bool binning=false, long n_bins=30) : KernelEstimator(kernel, binning, n_bins) {
        bw::check_estimator(bandwidth_estimator);
        _bandwidth_estimator = bandwidth_estimator;
    }

    /* initializes KDE object by specifying a kernel, and a bandwidth.
     * --- bdw : bandwidth for kernel (>= 0)
     * --- kernel : kernel to use. options are: gaussian, square, triangle, parabolic
     * --- binning : whether to bin the data or not.
     * --- n_bins : number of bins to use for BinData, if binning == false, n_bins is ignored. */
    explicit KDE(double bdw, const std::string& kernel="gaussian", bool binning=false, long n_bins=30) : KernelEstimator(bdw, kernel, binning, n_bins) {}

    /* fits density by storing data, and computing bandwidth by method specified during KDE construction. */
    void fit(const arma::mat& x) override;
    arma::vec fit_predict(const arma::mat& x) override;

    /* computes density for each point */
    arma::vec predict(const arma::mat& x) const override;

    /* samples density. */
    arma::vec sample(int n) const;
};

/* solves the lasso linear regression problem ||y-X*w||_2^2 + lambda*||w||_1 using coordinate descent. No intercept term, all coefficients are penalized.
 * --- w : weights/linear coefficients
 * --- X : independent variable.
 * --- y : dependent variable.
 * --- lambda : (> 0) L1 regularization parameter. Typical in the range 10^-3 for weak regularization to 10^3 for strong regularization
 * --- tol : tolerance for convergence, coordinate descent stops when |x_old - x_new| < tol.
 * --- max_iter : maximum number of iterations before premature stopping.
 * --- verbose : [false] no printing, [true] print each iteration.
 * returns [0] successful convergence [1] maximum iterations reach. */
int coordinate_lasso(arma::mat& w, const arma::mat& X, const arma::mat& y, double lambda, double tol=1e-4, u_long max_iter=1000, bool verbose=false);

/* solves the lasso linear regression problem ||y-(X*w + b)||_2^2 + lambda*||w||_1 using coordinate descent. Including intercept term, which is not penalized.
 * --- b : bias/intercept
 * --- w : weights/linear coefficients
 * --- X : independent variable.
 * --- y : dependent variable.
 * --- lambda : (> 0) L1 regularization parameter. Typical in the range 10^-3 for weak regularization to 10^3 for strong regularization
 * --- tol : tolerance for convergence, coordinate descent stops when |x_old - x_new| < tol.
 * --- max_iter : maximum number of iterations before premature stopping.
 * --- verbose : [false] no printing, [true] print each iteration.
 * returns [0] successful convergence [1] maximum iterations reach. */
int coordinate_lasso(arma::rowvec& b, arma::mat& w, const arma::mat& X, const arma::mat& y, double lambda, double tol=1e-4, long max_iter=1000, bool verbose=false);

class LinearModel : public Regressor {
    protected:
    arma::vec _w;
    double _b;
    double _lambda, _df;
    bool _fit_intercept;

    void _split_weights() {
        _b = _w(0);
        _w = _w.subvec(1, _w.n_elem-1);
    }
    arma::mat _add_intercept(const arma::mat& x) {
        arma::mat P(x.n_rows, _dim + 1);
        P.col(0).ones();
        P.tail_cols(x.n_cols) = x;
        return P;
    }

    public:
    const arma::vec& linear_coefs;
    const double& intercept;
    const double& lambda;
    const double& eff_df;

    explicit LinearModel(bool intercept) : linear_coefs(_w), intercept(_b), lambda(_lambda), eff_df(_df) {
        _fit_intercept = intercept;
    }
    void fit(const arma::mat& X, const arma::vec& y) = 0;
    arma::vec predict(const arma::mat& X) const = 0;
    double score(const arma::mat& X, const arma::vec& y) const = 0;
};

class LassoCV : public LinearModel {
    protected:
    double _tol;
    uint _max_iter;

    public:
    /* lasso_cv(tol=1e-5, max_iter=1000) : initialize cross validating lasso regression object.
     * --- intercept : whether to include intercept.
     * --- tol : tolerance for convergence, coordinate descent stops when |x_old - x_new| < tol.
     * --- max_iter : maximum number of iterations before premature stopping. */
    explicit LassoCV(bool intercept=true, double tol=1e-5, long max_iter=1000) : LinearModel(intercept) {
        if (tol < 0) {
            throw std::invalid_argument("require tol (=" + std::to_string(tol) + ") >= 0");
        }
        _tol = tol;
        if (max_iter < 1) {
            throw std::invalid_argument("require max_iter (=" + std::to_string(max_iter) + ") >= 1");
        }
        _max_iter = max_iter;
    }

    /* fit(X, y) : fit by cross-validation which benefits from a "warm-start" for the paramter estimates using the solution from the previous evaluation.
     * --- X : independent variable.
     * --- y : dependent variable. */
    void fit(const arma::mat& X, const arma::vec& y) override;
    arma::vec predict(const arma::mat& X) const override;

    /* returns R^2 score. */
    double score(const arma::mat& X, const arma::vec& y) const override;
};

class RidgeCV : public LinearModel {
    private:
    arma::mat _eigvecs;
    arma::vec _eigvals;

    public:
    const arma::mat& cov_eigvecs;
    const arma::vec& cov_eigvals;

    explicit RidgeCV(bool intercept=true) : LinearModel(intercept), cov_eigvecs(_eigvecs), cov_eigvals(_eigvals) {}

    /* fit(X,y) : fit a ridge regression model using the generalized formula for LOOCV.
     * --- X : indpendent variable data
     * --- y : dependent variable data. */
    void fit(const arma::mat& x, const arma::vec& y) override;
    arma::vec predict(const arma::mat& x) const override;
    double score(const arma::mat& x, const arma::vec& y) const override;
};

class Splines : public LinearModel {
    protected:
    arma::vec _c;
    arma::mat _X;
    arma::mat _eigvecs;
    arma::vec _eigvals;
    bool _fitted, _use_df, _use_lambda;

    public:
    const arma::vec& kernel_coefs;
    const arma::mat& X;
    const arma::mat& kernel_eigvecs;
    const arma::vec& kernel_eigvals;

    explicit Splines() : LinearModel(true), kernel_coefs(_c), X(_c), kernel_eigvecs(_eigvecs), kernel_eigvals(_eigvals) {
        _fitted = false;
        _use_df = false;
        _use_lambda = false;
    }

    void set_lambda(double l);
    void set_df(double df);

    void fit(const arma::mat& x, const arma::vec& y) override;
    arma::vec predict(const arma::mat& x) const override;
    double score(const arma::mat& x, const arma::vec& y) const override;    
};

inline void softmax_inplace(arma::mat& p) {
    p.each_row([](arma::rowvec& r) -> void {
        r -= r.max();
        r = arma::exp(r);
        r /= arma::accu(r);
    });
}

inline arma::mat softmax(const arma::mat& x) {
    arma::mat p = x;
    softmax_inplace(p);
    return p;
}

class LogisticRegression : public Classifier {
    protected:
    arma::rowvec _b;
    arma::mat _w;
    double _lambda;
    OneHotEncoder _encoder;
    
    public:
    const double& lambda;
    const arma::mat& linear_coefs;
    const arma::rowvec& intercepts;
    const OneHotEncoder& encoder;

    explicit LogisticRegression() : lambda(_lambda), linear_coefs(_w), intercepts(_b), encoder(_encoder) {
        _lambda = -1;
    }

    void set_lambda(double l) {
        if (l < 0) {
            throw std::invalid_argument("require lambda (=" + std::to_string(l) + ") >= 0");
        }
        _lambda = l;
    }

    void fit(const arma::mat& x, const arma::uvec& y) override;
    arma::uvec predict(const arma::mat& x) const override;
    arma::mat predict_proba(const arma::mat& x) const;
    double score(const arma::mat& x, const arma::uvec& y) const override;
};

class ModelSGD {
    protected:
    neuralnet::Model _mdl;
    neuralnet::fit_parameters _fitp;

    public:
    explicit ModelSGD(const std::string& loss, long max_iter, double tol, double l2, double l1, const std::string& optimizer, bool verbose) {
        _mdl.set_l2(l2);
        _mdl.set_l1(l1);
        _mdl.set_loss(loss);
        _mdl.set_optimizer(optimizer);
        _fitp.tol = tol;
        _fitp.max_iter = max_iter;
        _fitp.verbose = verbose;
    }
};

class LinearRegressorSGD : public Regressor, public ModelSGD {
    public:
    /* initializes linear regression model trained using stochastic gradient descent.
     * --- loss : loss function, choice of "mse", "mae".
     * --- max_iter : maximum number of iterations for optimization algorithm.
     * --- tol : stopping tolerance for optimization algorithm.
     * --- l2 : l2-norm regularization for coefficients, larger values induce a stronger penalization.
     * --- l1 : l1-norm regularization for coefficients, larger values induce a stronger penalization.
     * --- optimizer : optimization algorithm, choice of "adam", "sgd".
     * --- verbose : communicate solver progress. */
    explicit LinearRegressorSGD(const std::string& loss="mse", long max_iter=200, double tol=1e-4, double l2=1e-4, double l1=0, const std::string& optimizer="adam",bool verbose=false) : ModelSGD(loss,max_iter,tol,l2,l1,optimizer,verbose) {}

    void fit(const arma::mat& x, const arma::vec& y) override {
        _check_xy(x,y);
        _dim = x.n_cols;
        neuralnet::Layer lyr(_dim, 1);
        _mdl.attach(lyr);
        _mdl.compile();
        _mdl.fit(x,y,_fitp);
    }

    arma::vec predict(const arma::mat& x) const override {
        _check_x(x);
        return _mdl.predict(x);
    }

    double score(const arma::mat& x, const arma::vec& y) const override {
        _check_xy(x,y);
        return r2_score(y, predict(x));
    }
};

class LinearClassifierSGD : public Classifier, public ModelSGD {
    protected:
    OneHotEncoder _encoder;
    u_long _n_classes;
    
    public:
    /* initializes linear regression model trained using stochastic gradient descent.
     * --- loss : loss function, choice of "categorical_crossentropy".
     * --- max_iter : maximum number of iterations for optimization algorithm.
     * --- tol : stopping tolerance for optimization algorithm.
     * --- l2 : l2-norm regularization for coefficients, larger values induce a stronger penalization.
     * --- l1 : l1-norm regularization for coefficients, larger values induce a stronger penalization.
     * --- optimizer : optimization algorithm, choice of "adam", "sgd".
     * --- verbose : communicate solver progress. */
    explicit LinearClassifierSGD(const std::string& loss="categorical_crossentropy", long max_iter=200, double tol=1e-4, double l2=1e-4, double l1=0, const std::string& optimizer="adam",bool verbose=false) : ModelSGD(loss,max_iter,tol,l2,l1,optimizer,verbose) {}

    void fit(const arma::mat& x, const arma::uvec& y) override {
        _check_xy(x,y);
        _dim = x.n_cols;
        _encoder.fit(y);
        arma::mat onehot = _encoder.encode(y);
        _n_classes = onehot.n_cols;
        neuralnet::Layer lyr(_dim, _n_classes);
        lyr.set_activation("softmax");
        _mdl.attach(lyr);
        _mdl.compile();
        std::cout << _fitp.max_iter << "\n";
        _mdl.fit(x, onehot, _fitp);
    }

    arma::mat predict_proba(const arma::mat& x) const {
        _check_x(x);
        return _mdl.predict(x);
    }

    arma::uvec predict(const arma::mat& x) const override {
        return _encoder.decode(predict_proba(x));
    }

    double score(const arma::mat& x, const arma::uvec& y) const override {
        _check_xy(x,y);
        return accuracy_score(y, predict(x));
    }
};

class NeuralNetRegressor : public Regressor, public ModelSGD {
    protected:
    std::vector<std::pair<int,std::string>> _layers;

    public:
    /* initializes linear regression model trained using stochastic gradient descent.
     * --- layers : vector specifying hidden layers (not including output layer) where layers.at(i).first is the number of units in the i^th layer, and layers.at(i).second is the activation function for i^th layer. Choices of activation functions are found neuralnet
     * --- loss : loss function, choice of "mse", "mae".
     * --- max_iter : maximum number of iterations for optimization algorithm.
     * --- tol : stopping tolerance for optimization algorithm.
     * --- l2 : l2-norm regularization for coefficients, larger values induce a stronger penalization.
     * --- l1 : l1-norm regularization for coefficients, larger values induce a stronger penalization.
     * --- optimizer : optimization algorithm, choice of "adam", "sgd".
     * --- verbose : communicate solver progress. */
    explicit NeuralNetRegressor(const std::vector<std::pair<int,std::string>>& layers={{100,"relu"}}, const std::string& loss="mse", long max_iter=200, double tol=1e-4, double l2=1e-4, double l1=0, const std::string& optimizer="adam", bool verbose=false) : ModelSGD(loss,max_iter,tol,l2,l1,optimizer,verbose) {
        _layers = layers;
    }

    void fit(const arma::mat& x, const arma::vec& y) override {
        _check_xy(x,y);
        _dim = x.n_cols;
        long prev_dim = _dim;
        for (const std::pair<int,std::string>& l : _layers) {
            neuralnet::Layer lyr(prev_dim, l.first);
            lyr.set_activation(l.second);
            _mdl.attach(lyr);
            prev_dim = l.first;
        }
        neuralnet::Layer lyr(prev_dim, 1);
        _mdl.attach(lyr);
        _mdl.compile();
        _mdl.fit(x, y, _fitp);
    }

    arma::vec predict(const arma::mat& x) const override {
        _check_x(x);
        return _mdl.predict(x);
    }

    double score(const arma::mat& x, const arma::vec& y) const override {
        _check_xy(x,y);
        return r2_score(y, predict(x));
    }
};

class NeuralNetClassifier : public Classifier, public ModelSGD {
    protected:
    std::vector<std::pair<int,std::string>> _layers;
    OneHotEncoder _encoder;
    u_long _n_classes;

    public:
    /* initializes linear regression model trained using stochastic gradient descent.
     * --- layers : vector specifying hidden layers (not including output layer) where layers.at(i).first is the number of units in the i^th layer, and layers.at(i).second is the activation function for i^th layer. Choices of activation functions are found neuralnet
     * --- loss : loss function, choice of "categorical_crossentropy".
     * --- max_iter : maximum number of iterations for optimization algorithm.
     * --- tol : stopping tolerance for optimization algorithm.
     * --- l2 : l2-norm regularization for coefficients, larger values induce a stronger penalization.
     * --- l1 : l1-norm regularization for coefficients, larger values induce a stronger penalization.
     * --- optimizer : optimization algorithm, choice of "adam", "sgd".
     * --- verbose : communicate solver progress. */
    explicit NeuralNetClassifier(const std::vector<std::pair<int,std::string>>& layers={{100,"relu"}}, const std::string& loss="categorical_crossentropy", long max_iter=200, double tol=1e-4, double l2=1e-4, double l1=0, const std::string& optimizer="adam", bool verbose=false) : ModelSGD(loss,max_iter,tol,l2,l1,optimizer,verbose) {
        _layers = layers;
    }

    void fit(const arma::mat& x, const arma::uvec& y) override {
        _check_xy(x,y);
        _dim = x.n_cols;
        _encoder.fit(y);
        arma::mat onehot = _encoder.encode(y);
        _n_classes = onehot.n_cols;
        long prev_dim = _dim;
        for (const std::pair<int,std::string>& l : _layers) {
            neuralnet::Layer lyr(prev_dim, l.first);
            lyr.set_activation(l.second);
            _mdl.attach(lyr);
            prev_dim = l.first;
        }
        neuralnet::Layer lyr(prev_dim, _n_classes);
        lyr.set_activation("softmax");
        _mdl.attach(lyr);
        _mdl.compile();
        _mdl.fit(x, onehot, _fitp);
    }

    arma::mat predict_proba(const arma::mat& x) const {
        _check_x(x);
        return _mdl.predict(x);
    }

    arma::uvec predict(const arma::mat& x) const override {
        return _encoder.decode(predict_proba(x));
    }

    double score(const arma::mat& x, const arma::uvec& y) const override {
        _check_xy(x,y);
        return accuracy_score(y, predict(x));
    }
};

#endif