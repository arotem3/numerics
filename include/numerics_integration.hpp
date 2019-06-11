double integrate(const dfunc&, double, double, integrator i = LOBATTO, double err = 1e-5);
double simpson_integral(const dfunc&, double, double, double err = 1e-5);
double lobatto_integral(const dfunc&, double, double, double err = 1e-5);
double chebyshev_integral(const dfunc&, double, double, uint n = 25);

double mcIntegrate(const vec_dfunc&, const arma::vec&, const arma::vec&, double err = 1e-2, int N = 1e3);