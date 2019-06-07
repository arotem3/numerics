// --- difference methods ----- //
    arma::vec jacobian_diag(const vector_func&, const arma::vec&);

    void approx_jacobian(const vector_func&, arma::mat&, const arma::vec&, double err = 1e-2, bool catch_zero = true);

    arma::vec grad(const vec_dfunc&, const arma::vec&, double err = 1e-5, bool catch_zero = true);
    
    double deriv(const dfunc&, double, double err = 1e-5, bool catch_zero = true);

    polyInterp specral_deriv(const dfunc&, double a, double b, uint sample_points = 50);