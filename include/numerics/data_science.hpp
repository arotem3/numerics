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
    private:
    int k;              // number of clusters
    uint dim;           // dimension of problem space
    uint num_iters;     // number of iterations needed to converge
    arma::mat C;        // each column is a cluster mean
    arma::mat data;     // the data set ~ this is a pointer to the data set, so it will not be deleted when kmeans is deleted!
    arma::vec dataCluster;       // the i^th elem of dataCluster is the cluster number for the i^th data input. 
    int closest_cluster(const arma::rowvec&); // find closest cluster to a data pt
    arma::mat init_clusters();

    public:
    uint max_iterations;
    kmeans(arma::mat& x, int k, int max_iter = 100);
    kmeans(std::istream&);
    void load(std::istream&);
    void save(std::ostream&);

    arma::vec get_clusters() const;
    arma::mat get_centroids() const;

    arma::vec operator()(const arma::mat&);
    arma::vec predict(const arma::mat&);
    int operator()(const arma::rowvec&);
    int predict(const arma::rowvec&);

    arma::mat operator[](uint);
    arma::mat all_from_cluster(uint);

    std::ostream& summary(std::ostream& out = std::cout);
    std::ostream& help(std::ostream& out = std::cout);
};

class splines {
    private:
    arma::mat c, d;
    arma::mat X, Y;
    arma::vec cv_scores;
    std::vector<std::vector<int>> monomials;
    int n, m, dim;
    double lambda, df, gcv;
    
    void gen_monomials();
    arma::mat rbf(const arma::mat& x, const arma::mat& xgrid);

    public:
    splines(int m);
    splines(double lambda = -1, int m = 1);
    splines(const arma::mat&, const arma::mat&, int m = 1);
    splines(const arma::mat&, const arma::mat&, double, int m);
    splines(std::istream&);

    splines& fit(const arma::mat&, const arma::mat&);
    arma::mat fit_predict(const arma::mat&, const arma::mat&);

    arma::mat predict(const arma::mat&);
    arma::mat operator()(const arma::mat&);

    arma::mat data_X();
    arma::mat data_Y();

    arma::mat rbf(const arma::mat&);
    arma::mat polyKern(const arma::mat&);
    arma::mat poly_coef() const;
    arma::mat rbf_coef() const;

    double gcv_score() const;
    double eff_df() const;
    double smoothing_param() const;

    void load(std::istream&);
    void save(std::ostream&);
};

class kernel_smooth {
    private:
    arma::vec x, y;
    bin_data bins;
    int n;
    double bdw;
    kernels kern;
    bool binning;
    // arma::vec predict(const arma::vec&, const arma::vec&, const arma::vec&, double h);

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

class regularizer {
    private:
    arma::mat coefs, regular_mat;
    double lambda, cv, df;
    bool use_L2, use_cgd;
    void cross_validate(const arma::mat&, const arma::mat&);
    void fit_no_replace(const arma::mat&, const arma::mat&, double);

    public:
    regularizer(double lambda = arma::datum::nan);
    regularizer(const arma::mat&, double lambda = arma::datum::nan);
    regularizer(const arma::mat&, const arma::mat&, double lambda = arma::datum::nan);
    regularizer(const arma::mat&, const arma::mat&, const arma::mat&, double lambda = arma::datum::nan, bool use_conj_grad = true);

    arma::mat fit(const arma::mat&, const arma::mat&, bool use_conj_grad = true);

    arma::mat regularizing_mat() const;
    arma::mat coef();

    double MSE() const;
    double eff_df() const;
    double regularizing_param() const;
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