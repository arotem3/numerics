// --- data analysis ---------- //
enum class kernels {
    gaussian,
    square,
    triangle,
    parabolic
};

enum class bandwidth_estimator {
    rule_of_thumb_sd,
    min_sd_iqr,
    direct_plug_in,
    grid_cv
};

enum class knn_algorithm {
    AUTO,
    KD_TREE,
    BRUTE
};

enum class knn_metric {
    CONSTANT,
    L1_DISTANCE,
    L2_DISTANCE
};

class k_folds {
    private:
    int direction, num_folds;
    arma::mat X, Y;
    arma::umat I, range;

    public:
    k_folds(const arma::mat&, const arma::mat&, uint k=2, uint dim=0);
    arma::mat train_set_X(uint);
    arma::mat train_set_Y(uint);
    arma::mat test_set_X(uint);
    arma::mat test_set_Y(uint);
};

class k_folds_1d {
    private:
    int direction, num_folds;
    arma::mat X;
    arma::umat I, range;

    public:
    k_folds_1d(const arma::mat&, uint k=2, uint dim=0);
    arma::mat train_set(uint);
    arma::mat test_set(uint);
};

class bin_data {
    private:
    int _n;
    double _bin_width;
    arma::vec _bins;
    arma::vec _counts;

    public:
    const int& n_bins;
    const double& bin_width;
    const arma::vec& bins;
    const arma::vec& counts;

    bin_data(uint bins=0) : n_bins(_n), bin_width(_bin_width), bins(_bins), counts(_counts) {
        _n = bins;
    };
    
    void to_bins(const arma::vec& x, const arma::vec& y);
    void to_bins(const arma::vec& x);
};

class bins_nd {

};

class kmeans {
    protected:
    arma::mat _clusters, _intra_dist, _nearest_point;
    arma::uvec _nearest_idx;
    uint _k, _max_iter, _p, _dim;
    void _update_labels(arma::uvec& labels, arma::vec& nearest_dist, double& sum_p, const arma::mat& data, uint m);
    void _update_labels(arma::uvec& labels, const arma::mat& data);
    void _update_labels_nearest(arma::uvec& labels, const arma::mat& data);
    void _init_clusters(arma::uvec& labels, const arma::mat& data);
    void _update_intra_dist();
    double _d(const arma::rowvec& a, const arma::rowvec& b);

    public:
    const arma::mat& clusters;
    const arma::mat& cluster_distances;
    const arma::mat& points_nearest_centers;
    const arma::uvec& index_nearest_centers;
    kmeans(uint k, uint p_norm=2, uint max_iter=100);

    arma::uvec fit(const arma::mat& data, double tol=1e-2);

    arma::uvec predict(const arma::mat& data);
    arma::uvec operator()(const arma::mat& data);
};

class kmeans_sgd : public kmeans {
    private:
    void _update_batch_labels(arma::uvec& labels, const arma::mat& data, const arma::uvec& p, uint i, uint f);
    void _sgd_steps(arma::uvec& labels, const arma::mat& data, uint batch_size, uint max_iter);

    public:
    kmeans_sgd(uint k, uint p_norm=2);

    arma::uvec fit(const arma::mat& data, uint batch_size=100, uint max_iter=100);
};

class splines {
    private:
    arma::mat _c, _d;
    arma::mat _X, _Y;
    arma::mat _res;
    arma::mat _eigvecs;
    arma::vec _eigvals;
    std::vector<std::vector<uint>> _monomials;
    uint _deg, _dim;
    double _lambda, _df, _rmse;
    bool _fitted, _use_df;
    
    void gen_monomials();
    arma::mat eval_rbf();

    public:
    const double& smoothing_param;
    const double& eff_df;
    const double& RMSE;
    const arma::mat& residuals;
    const arma::mat& poly_coef;
    const arma::mat& rbf_coef;
    const arma::mat& data_X;
    const arma::mat& data_Y;
    const arma::vec& rbf_eigenvals;
    const arma::mat& rbf_eigenvecs;

    splines(uint poly_degree=1);
    splines(const std::string& s);

    void set_smoothing_param(double);
    void set_degrees_of_freedom(double);

    void fit(const arma::mat&, const arma::mat&);

    arma::mat predict(const arma::mat&);
    arma::mat operator()(const arma::mat&);

    arma::mat eval_rbf(const arma::mat&);
    arma::mat eval_poly(const arma::mat&);

    void load(const std::string& s);
    void save(const std::string& s="model.splines");
};

class kernel_smooth {
    private:
    arma::vec x, y;
    bin_data bins;
    int n;
    double bdw;
    kernels kern;
    bool binning;

    public:
    const arma::vec& data_x;
    const arma::vec& data_y;
    const double& bandwidth;

    kernel_smooth(kernels k=kernels::gaussian, bool binning=false);
    kernel_smooth(double bdw, kernels k=kernels::gaussian, bool binning=false);
    kernel_smooth(const std::string&);

    void save(const std::string& fname="kern_smooth.KS");
    void load(const std::string&);

    kernel_smooth& fit(const arma::vec&, const arma::vec&);
    arma::vec fit_predict(const arma::vec&, const arma::vec&);
    
    double predict(double);
    arma::vec predict(const arma::vec&);
    double operator()(double);
    arma::vec operator()(const arma::vec&);
};

namespace bw { // a selection of bandwidth estimation methods
    arma::vec eval_kernel(const arma::vec& x, numerics::kernels K);
    double dpi(const arma::vec& x, double s=0, numerics::kernels K=numerics::kernels::gaussian);
    double dpi_binned(const numerics::bin_data& bins, double s=0, numerics::kernels K=numerics::kernels::gaussian);
    double rot1(int n, double s);
    double rot2(const arma::vec& x, double s = 0);
    double grid_mse(const arma::vec& x, numerics::kernels K, double s=0, int grid_size=20, bool binning=false);
};

class kde {
    private:
    arma::vec x;
    bin_data bins;
    double bdw, stddev;
    kernels kern;
    bandwidth_estimator method;
    bool binning;

    public:
    const double& bandwidth;
    const arma::vec& data;


    kde(kernels k=kernels::gaussian, bandwidth_estimator method = bandwidth_estimator::min_sd_iqr, bool binning = false);
    kde(double bdw, kernels k=kernels::gaussian, bool binning = false);
    kde(const std::string&);

    void save(const std::string& fname="kde.kde");
    void load(const std::string&);

    kde& fit(const arma::vec&);

    arma::vec sample(uint n=1);
    
    double predict(double);
    arma::vec predict(const arma::vec&);
    double operator()(double);
    arma::vec operator()(const arma::vec&);
};

int coordinate_lasso(const arma::mat& y, const arma::mat& X, arma::mat& w, double lambda, bool first_term_intercept, double tol, uint max_iter, bool verbose=false);

class lasso_cv {
    double _lambda, _tol, _rmse;
    uint _max_iter, _df;
    arma::mat _w;
    arma::mat _res;

    public:
    const double& regularizing_param;
    const double& RMSE;
    const arma::mat& coef;
    const arma::mat& residuals;
    const uint& eff_df;

    lasso_cv(double tol=1e-5, uint max_iter=1000);
    void fit(const arma::mat& X, const arma::mat& y, bool first_term_intercept=false);
};

class ridge_cv {
    private:
    arma::mat _w;
    arma::mat _res;
    arma::mat _eigvecs;
    arma::vec _eigvals;
    double _lambda, _rmse, _df;

    public:
    const arma::mat& coef;
    const arma::mat& residuals;
    const arma::mat& cov_eigvecs;
    const arma::vec& cov_eigvals;
    const double& regularizing_param;
    const double& RMSE;
    const double& eff_df;

    ridge_cv();

    void fit(const arma::mat&, const arma::mat&);
};

class logistic_regression {
    private:
    arma::mat x,y,c,d;
    arma::vec L, cv_scores;
    double lambda, beta;
    int n_obs;
    arma::mat softmax(const arma::mat&);
    void fit_linear(double lam);
    void fit_no_replace(const arma::mat& X, const arma::mat& Y, double lam);

    public:
    logistic_regression(double Beta = 1, double Lambda = arma::datum::nan);
    logistic_regression(std::istream& in);
    void load(std::istream& in);
    void save(std::ostream& out);

    void fit(const arma::mat& X, const arma::mat& Y);

    arma::mat rbf(const arma::mat& xgrid);
    arma::mat predict_probabilities(const arma::mat& xgrid);
    arma::umat predict_categories(const arma::mat& xgrid);

    double regularizing_param() const {
        return lambda;
    }
    double kernel_param() const {
        return beta;
    }
    arma::mat linear_coefs() const {
        return c;
    }
    arma::mat kernel_coefs() const {
        return d;
    }
    arma::mat get_cv_results() const {
        arma::mat rslts = arma::zeros(L.n_rows,2);
        rslts.col(0) = L;
        rslts.col(1) = cv_scores;
        return rslts;
    }
    arma::mat data_X() {
        return x;
    }
    arma::mat data_Y() {
        return y;
    }
};

class knn_regression {
    protected:
    bool categorical_loss;
    numerics_private_utility::kd_tree_util::kd_tree X_tree;
    arma::mat X_array, Y;
    arma::uvec kk;
    int k;
    knn_algorithm alg;
    knn_metric metr;
    double score_regression(const arma::mat& train_X, const arma::mat& train_Y, const arma::mat& test_X, const arma::mat& test_Y, int K);
    arma::rowvec voting_func(const arma::rowvec& pt, const arma::mat& neighbors_X, const arma::mat& neighbors_Y);
    arma::uvec brute_knn(const arma::rowvec& pt, const arma::mat& X, int K);

    public:
    const int& num_neighbors;
    const arma::mat& data_X;
    const numerics_private_utility::kd_tree_util::kd_tree& tree_X;
    const arma::mat& data_Y;

    knn_regression(uint K, knn_algorithm algorithm = knn_algorithm::AUTO, knn_metric metric = knn_metric::CONSTANT);
    knn_regression(const arma::uvec K_set, knn_algorithm algorithm = knn_algorithm::AUTO, knn_metric metric = knn_metric::CONSTANT);
    void fit(const arma::mat& X, const arma::mat& Y);
    arma::mat predict(const arma::mat& xgrid);
    arma::mat operator()(const arma::mat& xgrid);
};

class knn_classifier : private knn_regression {
    private:
    arma::uvec cats;

    public:
    const arma::uvec& categories;
    using knn_regression::num_neighbors;
    using knn_regression::data_X;
    using knn_regression::tree_X;

    knn_classifier(uint K, knn_algorithm algorithm = knn_algorithm::AUTO, knn_metric metric = knn_metric::CONSTANT);
    knn_classifier(const arma::uvec K_set, knn_algorithm algorithm = knn_algorithm::AUTO, knn_metric metric = knn_metric::CONSTANT);
    
    void fit(const arma::mat& X, const arma::uvec& y);
    
    arma::mat predict_probabilities(const arma::mat& xgrid);
    arma::uvec predict_categories(const arma::mat& xgrid);
};