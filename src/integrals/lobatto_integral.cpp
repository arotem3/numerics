#include <numerics.hpp>

/* lobatto_integral(f, a, b, tol) : adaptive gauss-Lobato's method, spectrally accurate for smooth functions
 * --- f : function to integrate.
 * --- a,b : interval [a,b] to evaluate integral over
 * --- tol : error tolerance, i.e. stopping criterion */
double numerics::lobatto_integral(const std::function<double(double)>& f, double a, double b, double tol) {
    tol = std::abs(tol);
    double h = (b-a)/2;
    double c = (b+a)/2;

    double sum4(0), sum7(0);
    for (int i(0); i < 7; i++) {
        if (i < 4) {
            sum4 += numerics_private_utility::W4[i] * f(h * numerics_private_utility::X4[i] + c);
        }
        sum7 += numerics_private_utility::W7[i] * f(h * numerics_private_utility::X7[i] + c);
    }
    sum4 *= h;
    sum7 *= h;

    if (std::abs(sum4 - sum7) < tol) return sum4;
    else return lobatto_integral(f,a,c,tol/2) + lobatto_integral(f,c,b,tol/2);
}