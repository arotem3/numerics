// --- root finding ----------- //
    //--- linear ---//
    void cgd(arma::mat&, arma::mat&, arma::mat&, cg_opts&);
    cg_opts cgd(arma::mat&, arma::mat&, arma::mat&);

    void sp_cgd(const arma::sp_mat&, const arma::mat&, arma::mat&, cg_opts&);
    cg_opts sp_cgd(const arma::sp_mat&, const arma::mat&, arma::mat&);

    //--- nonlinear ---//

    void newton(const vector_func&, const vec_mat_func&, arma::vec&, nonlin_opts& opts);
    nonlin_opts newton(const vector_func&, const vec_mat_func&, arma::vec&);
    
    void broyd(const vector_func&, arma::vec&, nonlin_opts&);
    nonlin_opts broyd(const vector_func&, arma::vec&);

    void lmlsqr(const vector_func&, arma::vec&, lsqr_opts&);
    lsqr_opts lmlsqr(const vector_func&, arma::vec&);

    void mix_fpi(const vector_func&, arma::vec&, fpi_opts&);
    fpi_opts mix_fpi(const vector_func&, arma::vec&);
    
    //--- for optimization ---//
    void newton(const vec_dfunc&, const vector_func&, const vec_mat_func&, arma::vec&, nonlin_opts& opts);

    void bfgs(const vec_dfunc&, const vector_func&, arma::vec&, nonlin_opts&);
    nonlin_opts bfgs(const vec_dfunc&, const vector_func&, arma::vec&);

    void lbfgs(const vec_dfunc&, const vector_func&, arma::vec&, lbfgs_opts&);
    lbfgs_opts lbfgs(const vec_dfunc&, const vector_func&, arma::vec&);

    void mgd(const vector_func&, arma::vec&, gd_opts&);
    gd_opts mgd(const vector_func&, arma::vec&);

    void sgd(const sp_vector_func&, arma::vec&, gd_opts&);
    gd_opts sgd(const sp_vector_func&, arma::vec&);

    void nlcgd(const vector_func&, arma::vec&, nonlin_opts&);
    nonlin_opts nlcgd(const vector_func&, arma::vec&);

    void adj_gd(const vector_func&, arma::vec&, nonlin_opts&);
    nonlin_opts adj_gd(const vector_func&, arma::vec&);

    //--- univariate ---//
    double fzero(const dfunc&, double, double);
    double newton(const dfunc&, const dfunc&, double, double err = 1e-10);
    double secant(const dfunc&, double, double);
    double bisect(const dfunc&, double, double, double tol = 1e-8);
// --- optimization ----------- //
    //--- unconstrained : generic ---//
    double minimize_unc(const vec_dfunc&, arma::vec&, optim_opts&);
    double minimize_unc(const vec_dfunc&, arma::vec&);

    //--- linear contrained ---//
    double simplex(arma::mat&, arma::vec&);
    double simplex(const arma::rowvec&, const arma::mat&, const arma::vec&, arma::vec&);

    //--- box contrained gradient free ---//
    double genOptim(const vec_dfunc&, arma::vec&, const arma::vec&, const arma::vec&, gen_opts&);
    double genOptim(const vec_dfunc&, arma::vec&, const arma::vec&, const arma::vec&);
    
    //--- unconstrained gradient free ---//
    double genOptim(const vec_dfunc&, arma::vec&, gen_opts&);
    double genOptim(const vec_dfunc&, arma::vec&);

    //--- integer boolean ---//
    double boolOptim(std::function<double(const arma::uvec&)>, arma::uvec&, int);