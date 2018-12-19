#include "numerics.hpp"

//--- computes the jacobian of a system of nonlinear equations ---//
//----- f  : f(x) whose jacobian to approximate ------------------//
//----- x  : vector to evaluate jacobian at ----------------------//
//----- err: approximate upper error bound -----------------------//
void numerics::approx_jacobian(std::function<arma::vec(const arma::vec&)> f, arma::mat& J, const arma::vec& x, double err, bool catch_zero) {
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

//--- computes the gradient of a function of multiple variables ---//
//----- f  : f(x) whose gradient to approximate -------------------//
//----- x  : vector to evaluate gradient at -----------------------//
//----- err: approximate upper error bound ------------------------//
arma::vec numerics::grad(std::function<double(const arma::vec&)> f, const arma::vec& x, double err, bool catch_zero) {
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

//--- computes the approximate derivative of a function of a single variable ---//
//----- f  : f(x) whose derivative to approximate ------------------------------//
//----- x  : point to evaluate derivative --------------------------------------//
//----- err: approximate upper error bound; method is O(h^4) -------------------//
double numerics::deriv(std::function<double(double)> f, double x, double err, bool catch_zero) {
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