#include <numerics.hpp>

/* GRAD : computes the gradient of a function of multiple variables.
 * --- f  : f(x) whose gradient to approximate.
 * --- x  : vector to evaluate gradient at.
 * --- h  : finite difference step size. method is O(h^4).
 * --- catch_zero: rounds near zero elements to zero. */
arma::vec numerics::grad(const std::function<double(const arma::vec&)>& f, const arma::vec& x, double h, bool catch_zero, short npt) {
    uint n = x.n_elem;
    arma::vec g(n);
    if (npt == 1) { // specialize this instance to minimize repeated calculations.
        g.fill(-f(x));
        arma::vec y = x;
        for (uint i=0; i < n; ++i) {
            y(i) += h;
            g(i) += f(y);
            y(i) = x(i);
        }
        g /= h;
        g(arma::find(arma::abs(g) < h/2)).zeros();
    } else {
        for (uint i=0; i < n; ++i) {
            auto ff = [&f,x,i](double t) -> double {
                arma::vec y = x;
                y(i) = t;
                return f(y);
            };
            g(i) = deriv(ff, x(i), h, catch_zero, npt);
        }
    }
    
    return g;
}