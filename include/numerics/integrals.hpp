#ifndef NUMERICS_INTEGRALS_HPP
#define NUMERICS_INTEGRALS_HPP

/* adaptive simpson's method, generally efficient.
 * --- fmap : if fmap is provided, all function evaluations will be stored here.
 * --- f : function to integrate.
 * --- a,b : interval [a,b] to evaluate integral over.
 * --- tol : error tolerance, i.e. stopping criterion */
double simpson_integral(std::map<double,double>& fmap, const std::function<double(double)>& f, double a, double b, double tol = 1e-5);

/* adaptive simpson's method, generally efficient.
 * --- f : function to integrate.
 * --- a,b : interval [a,b] to evaluate integral over.
 * --- tol : error tolerance, i.e. stopping criterion */
double simpson_integral(const std::function<double(double)>& f, double a, double b, double tol = 1e-5);

/* adaptive gauss-Lobato's method, spectrally accurate for smooth functions
 * --- f : function to integrate.
 * --- a,b : interval [a,b] to evaluate integral over
 * --- tol : error tolerance, i.e. stopping criterion */
double lobatto_integral(const std::function<double(double)>& f, double a, double b, double tol = 1e-8);

/* polynomial interpolation based integration, spectral method for integrating continuous functions 
 * --- f : function to integrate.
 * --- a,b : interval [a,b] to evaluate integral over
 * --- m : exact number of function evaluations. method is O(m log m) in time (via the fft).  */
double chebyshev_integral(const std::function<double(double)>& f, double a, double b, uint m = 32);

/* general integration method.
 * --- f : function to integrate.
 * --- a,b : interval [a,b] to evaluate integral over
 * --- method : one of {lobatto, simpson, chebyshev}, specify method. */
inline double integrate(const std::function<double(double)>& f, double a, double b, const std::string& method="lobatto") {
    double integral = 0;
    if (method == "lobatto") integral = lobatto_integral(f,a,b);
    else if (method == "simpson") integral = simpson_integral(f,a,b);
    else if (method == "chebyshev") integral = chebyshev_integral(f,a,b);
    else {
        throw std::invalid_argument("integration method (=" + method + ") must be one of {lobatto, simpson, chebyshev}");
    }
    return integral;
}

#endif