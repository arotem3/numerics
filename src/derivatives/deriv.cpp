#include <numerics.hpp>

/* deriv(f, x, h, catch_zero) : computes the approximate derivative of a function of a single variable.
 * --- f  : f(x) whose derivative to approximate.
 * --- x  : point to evaluate derivative.
 * --- h  : finite difference step size; method is O(h^4).
 * --- catch_zero: rounds near zero elements to zero. */
double numerics::deriv(const std::function<double(double)>& f, double x, double h, bool catch_zero, short npt) {
    double df;
    if (npt == 1) df = (f(x + h) - f(x))/h;
    else if (npt == 2) df = (f(x + h) - f(x - h))/(2*h);
    else if (npt == 4) df = (f(x - 2*h) - 8*f(x - h) + 8*f(x + h) - f(x + 2*h))/(12*h);
    else {
        throw std::invalid_argument("only 1, 2, and 4 point FD derivatives supported (not " + std::to_string(npt) + ").");
    }
    
    if (catch_zero and (std::abs(df) < h/2)) return 0;

    return df;
}