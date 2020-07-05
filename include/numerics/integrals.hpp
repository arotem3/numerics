// --- integration
enum class integrator {
    SIMPSON,
    LOBATTO
};

double simpson_integral(const std::function<double(double)>& f, double a, double b, std::map<double,double>& fmap, double tol = 1e-5);
double simpson_integral(const std::function<double(double)>& f, double a, double b, double tol = 1e-5);

double lobatto_integral(const std::function<double(double)>& f, double a, double b, double tol = 1e-5);

double chebyshev_integral(const std::function<double(double)>& f, double a, double b, uint m = 32);

inline double integrate(const std::function<double(double)>& f, double a, double b, double tol = 1e-5, integrator i = integrator::LOBATTO) {
    if (i == integrator::SIMPSON) return simpson_integral(f,a,b,tol);
    else return lobatto_integral(f,a,b,tol);
}