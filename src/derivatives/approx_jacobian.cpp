#include <numerics.hpp>

/* approx_jacobian(f, x, h, catch_zero) : computes the jacobian of a system of nonlinear equations.
 * --- f  : f(x) whose jacobian to approximate.
 * --- x  : vector to evaluate jacobian at.
 * --- h : finite difference step size. method is O(h^4)
 * --- catch_zero: rounds near zero elements to zero. */
arma::mat numerics::approx_jacobian(const std::function<arma::vec(const arma::vec&)>& f, const arma::vec& x, double h, bool catch_zero) {
    uint m = f(x).n_elem; // num functions -> num rows
    uint n = x.n_elem; // num variables -> num cols
    arma::mat J = arma::zeros(m,n);
    for (uint i(0); i < m; ++i) {
        auto ff = [&f,i](const arma::vec& u) -> double {
            arma::vec z = f(u);
            return z(i);
        };
        J.row(i) = grad(ff, x, h, catch_zero).t();
    }
    return J;
}

/* jacobian_diag(f, x, h) : computes only the diagonal of a system of nonlinear equations.
 * --- f : f(x) system to approximate jacobian of.
 * --- x : vector to evaluate jacobian at.
 * --- h : finite difference step size. method is O(h^4) */
arma::vec numerics::jacobian_diag(const std::function<arma::vec(const arma::vec&)>& f, const arma::vec& x, double h) {
    arma::vec J = f(x - 2*h) - 8*f(x - h) + 8*f(x + h) - f(x + 2*h);
    J /= 12*h;
    return J;
}