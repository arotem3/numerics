// --- data analysis ---------- //
typedef enum KERNELS {
    RBF,
    square,
    triangle,
    parabolic
} kernels;

class k_folds {
    private:
    int direction, num_folds;
    arma::mat X, Y;
    arma::umat I, range;

    public:
    k_folds(const arma::mat&, const arma::mat&, uint k=2, uint dim=0);
    arma::mat fold_X(uint);
    arma::mat fold_Y(uint);
    arma::mat not_fold_X(uint);
    arma::mat not_fold_Y(uint);
    arma::mat operator[](int);
    arma::mat operator()(int);
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
    kmeans(arma::mat&, int);
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
    std::vector<std::vector<int>> monomials;
    int n, m, dim;
    double lambda, df, gcv;
    
    void gen_monomials();
    void fit(arma::mat&, arma::mat&, arma::mat&, arma::mat&);

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
    int n;
    double bdw, cv;
    kernels kern;
    arma::vec predict(const arma::vec&, const arma::vec&, const arma::vec&, double h);

    public:
    kernel_smooth(const arma::vec&, const arma::vec&, double bdw=0, kernels k=RBF);
    kernel_smooth(double bdw=0, kernels k=RBF);
    kernel_smooth(std::istream&);

    void save(std::ostream&);
    void load(std::istream&);

    kernel_smooth& fit(const arma::vec&, const arma::vec&);
    arma::vec fit_predict(const arma::vec&, const arma::vec&);
    
    double predict(double);
    arma::vec predict(const arma::vec&);
    double operator()(double);
    arma::vec operator()(const arma::vec&);
    
    arma::vec data_X();
    arma::vec data_Y();

    double bandwidth() const;
    double MSE() const;
};

class regularizer {
    private:
    arma::mat coefs, regular_mat;
    double lambda, cv, df;
    bool use_cgd, use_L2;
    void cross_validate(const arma::mat&, const arma::mat&);
    void fit_no_replace(const arma::mat&, const arma::mat&, double);

    public:
    regularizer(double lambda = arma::datum::nan);
    regularizer(const arma::mat&, double lambda = arma::datum::nan);
    regularizer(const arma::mat&, const arma::mat&, double lambda = arma::datum::nan, bool use_conj_grad = true);
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
    std::string fit_linear(double lam);
    std::string fit_no_replace(const arma::mat& X, const arma::mat& Y, double lam);

    public:
    logistic_regression(double Beta = 1, double Lambda = arma::datum::nan);
    logistic_regression(std::istream& in);
    void load(std::istream& in);
    void save(std::ostream& out);

    void fit(const arma::mat& X, const arma::mat& Y, bool echo = true);

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