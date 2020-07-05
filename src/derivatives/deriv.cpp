#include <numerics.hpp>

/* deriv(f, x, h, catch_zero) : computes the approximate derivative of a function of a single variable.
 * --- f  : f(x) whose derivative to approximate.
 * --- x  : point to evaluate derivative.
 * --- h  : finite difference step size; method is O(h^4).
 * --- catch_zero: rounds near zero elements to zero. */
double numerics::deriv(const std::function<double(double)>& f, double x, double h, bool catch_zero) {
    double df = f(x - 2*h) - 8*f(x - h) + 8*f(x + h) - f(x + 2*h);
    df /= 12*h;
    if (catch_zero && std::abs(df) < h) return 0; // helps if we expect sparse derivatives
    else return df;
}