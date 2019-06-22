// --- difference methods ----- //
arma::vec jacobian_diag(const std::function<arma::vec(const arma::vec&)>& f, const arma::vec& x, double h = 1e-2);

arma::mat approx_jacobian(const std::function<arma::vec(const arma::vec&)>& f, const arma::vec& x, double h=1e-2, bool catch_zero = true);

arma::vec grad(const std::function<double(const arma::vec&)>& f, const arma::vec& x, double h=1e-2, bool catch_zero = true);

double deriv(const std::function<double(double)>& f, double x, double h=1e-2, bool catch_zero = true);

poly_interp spectral_deriv(const std::function<double(double)>& f, double a, double b, uint sample_points = 50);