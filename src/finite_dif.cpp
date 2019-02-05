#include "numerics.hpp"

/* APPROX_JACOBIAN : computes the jacobian of a system of nonlinear equations.
     * --- f  : f(x) whose jacobian to approximate.
     * --- x  : vector to evaluate jacobian at.
     * --- err: approximate upper error bound.
     * --- catch_zero: rounds near zero elements to zero. */
void numerics::approx_jacobian(const vector_func& f, arma::mat& J, const arma::vec& x, double err, bool catch_zero) {
    size_t m = f(x).n_elem; // num functions -> num rows
    size_t n = x.n_elem; // num variables -> num cols
    J = arma::zeros(m,n);
    for (size_t i(0); i < m; ++i) {
        auto ff = [f,i](const arma::vec& u) -> double {
            arma::vec z = f(u);
            return z(i);
        };
        J.row(i) = grad(ff,x,err, catch_zero).t();
    }
}

/* JACOBIAN_DIAG : computes only the diagonal of a system of nonlinear equations.
 * --- f : f(x) system to approximate jacobian of.
 * --- x : vector to evaluate jacobian at. */
arma::vec numerics::jacobian_diag(const vector_func& f, const arma::vec& x) {
    double h = 1e-3;
    arma::vec J = f(x - 2*h) - 8*f(x - h) + 8*f(x + h) - f(x + 2*h);
    J /= 12*h;
    return J;
}

/* GRAD : computes the gradient of a function of multiple variables.
 * --- f  : f(x) whose gradient to approximate.
 * --- x  : vector to evaluate gradient at.
 * --- err: approximate upper error bound.
 * --- catch_zero: rounds near zero elements to zero. */
arma::vec numerics::grad(const vec_dfunc& f, const arma::vec& x, double err, bool catch_zero) {
    size_t n = x.n_elem;
    arma::vec g(n,arma::fill::zeros);
    for (size_t i(0); i < n; ++i) {
        auto ff = [f,x,i](double t) -> double {
            arma::vec y = x;
            y(i) = t;
            return f(y);
        };
        g(i) = deriv(ff,x(i),err, catch_zero);
    }
    return g;
}

/* DERIV : computes the approximate derivative of a function of a single variable.
 * --- f  : f(x) whose derivative to approximate.
 * --- x  : point to evaluate derivative.
 * --- err: approximate upper error bound; method is O(h^4).
 * --- catch_zero: rounds near zero elements to zero. */
double numerics::deriv(const dfunc& f, double x, double err, bool catch_zero) {
    double h = 1e-2;
    double df = f(x - 2*h) - 8*f(x - h) + 8*f(x + h) - f(x + 2*h);
    df /= 12*h;
    if (catch_zero && std::abs(df) < err) return 0; // helps if we expect sparse derivatives
    h *= 0.75;
    double df1 = f(x - 2*h) - 8*f(x - h) + 8*f(x + h) - f(x + 2*h);
    df1 /= 12*h;

    while (std::abs(df1 - df) > err) {
        df = df1;
        h *= 0.75;
        df1 = f(x - 2*h) - 8*f(x - h) + 8*f(x + h) - f(x + 2*h);
        df1 /= 12*h;
    }
    return df;
}