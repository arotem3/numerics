// --- data analysis ---------- //
    folds k_fold(const arma::mat&, const arma::mat&, uint k=2, uint dim=0);

    class kmeans {
        private:
        int k;              // number of clusters
        uint dim;           // dimension of problem space
        uint num_iters;     // number of iterations needed to converge
        arma::mat C;        // each column is a cluster mean
        arma::mat data;     // the data set ~ this is a pointer to the data set, so it will not be deleted when kmeans is deleted!
        arma::vec dataCluster;       // the i^th elem of dataCluster is the cluster number for the i^th data input. 
        int closestC(const arma::rowvec&); // find closest cluster to a data pt
        arma::mat init_clusters();

        public:
        kmeans(arma::mat&, int);
        kmeans(std::istream&);
        void load(std::istream&);
        void save(std::ostream&);

        arma::vec getClusters() const;
        arma::mat getCentroids() const;

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
        splines(const arma::mat&, const arma::mat&, double, int m = 1);
        splines(std::istream&);

        splines& fit(const arma::mat&, const arma::mat&);
        arma::mat fit_predict(const arma::mat&, const arma::mat&);

        arma::mat predict(const arma::mat&);
        arma::mat operator()(const arma::mat&);

        arma::mat data_X();
        arma::mat data_Y();

        arma::mat rbf(const arma::mat&);
        arma::mat polyKern(const arma::mat&);
        arma::vec poly_coef() const;
        arma::vec rbf_coef() const;

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
        arma::mat X, Y, y_hat, coefs, regular_mat;
        double lambda, cv, df;
        int n_obs, x_dim, y_dim;
        bool use_cgd, use_L2;
        void cross_validate();
        void fit_no_replace(const arma::mat&, const arma::mat&, double);

        public:
        regularizer(double lambda = arma::datum::nan);
        regularizer(const arma::mat&, double lambda = arma::datum::nan);
        regularizer(const arma::mat&, const arma::mat&, double lambda = arma::datum::nan, bool use_conj_grad = true);
        regularizer(const arma::mat&, const arma::mat&, const arma::mat&, double lambda = arma::datum::nan, bool use_conj_grad = true);
        regularizer(std::istream&);

        regularizer& fit(const arma::mat&, const arma::mat&, bool use_conj_grad = true);
        arma::mat fit_predict(const arma::mat&, const arma::mat&, bool use_conj_grad = true);

        arma::mat predict(const arma::mat&);
        arma::mat predict();
        arma::mat operator()(const arma::mat&);
        arma::mat operator()();

        arma::mat data_X();
        arma::mat data_Y();
        arma::mat regularizing_matrix();
        arma::mat coef();

        double MSE() const;
        double eff_df() const;
        double regularizing_param() const;

        void save(std::ostream&);
        void load(std::istream&);
    };