#include <numerics.hpp>

arma::vec numerics::directional_grad(const std::function<arma::vec(const arma::vec&)>& f, const arma::vec& x, const arma::vec& v, double h, bool catch_zero, short npt) {
    if (x.n_elem != v.n_elem) {
        throw std::invalid_argument(
            "cannot compute derivative of f at x in the direction v when x.n_elem (="
            + std::to_string(x.n_elem) + ") does not match v.n_elem (="
            + std::to_string(v.n_elem) + ")."
        );
    }

    arma::vec Jv;

    double C = h / arma::norm(v);
    if (npt == 1) Jv = (f(x+C*v) - f(x)) / C;
    else if (npt == 2) Jv = (f(x+C*v) - f(x-C*v)) / (2*C);
    else if (npt == 4) Jv = (f(x - 2*C*v) - 8*f(x - C*v) + 8*f(x + C*v) - f(x + 2*C*v)) / (12*C);
    else {
        throw std::invalid_argument("only 1, 2, and 4 point FD derivatives supported (not " + std::to_string(npt) + ").");
    }

    if (catch_zero) Jv(arma::find(arma::abs(Jv) < h/2)).zeros();

    return Jv;
}