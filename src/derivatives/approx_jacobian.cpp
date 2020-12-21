#include <numerics.hpp>

arma::mat _J1p(const std::function<arma::vec(const arma::vec&)>& f, const arma::vec& x, double h) {
    arma::mat J = arma::repmat(-f(x), 1, x.n_elem);
    arma::vec y = x;
    for (uint i=0; i < x.n_elem; ++i) {
        y(i) += h;
        J.col(i) += f(y);
        y(i) = x(i);
    }
    J /= h;
    return J;
}

arma::mat _J2p(const std::function<arma::vec(const arma::vec&)>& f, const arma::vec& x, double h) {
    arma::vec y = x;
    y(0) += h; // perform first step outside of loop for inferring size of f.
    arma::vec ff = f(y);
    y(0) = x(0) - h;
    ff -= f(y);
    y(0) = x(0);
    uint m = ff.n_elem;
    arma::mat J(m,x.n_elem);
    J.col(0) = ff;
    for (uint i=1; i < x.n_elem; ++i) {
        y(i) += h;
        J.col(i) = f(y);
        y(i) = x(i) - h;
        J.col(i) -= f(y);
        y(i) = x(i);
    }
    J /= 2*h;
    return J;
}

arma::mat _J4p(const std::function<arma::vec(const arma::vec&)>& f, const arma::vec& x, double h) {
    arma::vec y = x;
    y(0) += 2*h;
    arma::vec ff = -f(y);
    y(0) = x(0) + h;
    ff += 8*f(y);
    y(0) = x(0) - h;
    ff -= 8*f(y);
    y(0) = x(0) - 2*h;
    ff += f(y);
    y(0) = x(0);

    arma::mat J(ff.n_elem, x.n_elem);
    for (uint i=1; i < x.n_elem; ++i) {
        y(i) += 2*h;
        J.col(i) = -f(y);
        y(i) = x(i) + h;
        J.col(i) += 8*f(y);
        y(i) = x(i) - h;
        J.col(i) -= 8*f(y);
        y(i) = x(i) - 2*h;
        J.col(i) += f(y);
        y(i) = x(i);
    }
    J /= 12*h;
    return J;
}

/* approx_jacobian(f, x, h, catch_zero) : computes the jacobian of a system of nonlinear equations.
 * --- f  : f(x) whose jacobian to approximate.
 * --- x  : vector to evaluate jacobian at.
 * --- h : finite difference step size. method is O(h^4)
 * --- catch_zero: rounds near zero elements to zero. */
arma::mat numerics::approx_jacobian(const std::function<arma::vec(const arma::vec&)>& f, const arma::vec& x, double h, bool catch_zero, short npt) {
    uint n = x.n_elem; // num variables -> num cols
    if (x.n_elem < 1) throw std::invalid_argument("when computing the jacobian, require x.n_elem (=" + std::to_string(x.n_elem) + ") >= 1.");

    arma::mat J;
    if (npt == 1) J = _J1p(f, x, h);
    else if (npt == 2) J = _J2p(f, x, h);
    else if (npt == 4) J = _J4p(f, x, h);
    else {
        throw std::invalid_argument("only 1, 2, and 4 point derivatives supported (not " + std::to_string(npt) + ").");
    }

    if (catch_zero) J(arma::find(arma::abs(J) < h/2)).zeros();

    return J;
}

/* jacobian_diag(f, x, h) : computes only the diagonal of a system of nonlinear equations.
 * --- f : f(x) system to approximate jacobian of.
 * --- x : vector to evaluate jacobian at.
 * --- h : finite difference step size. method is O(h^4) */
arma::vec numerics::jacobian_diag(const std::function<arma::vec(const arma::vec&)>& f, const arma::vec& x, double h, bool catch_zero, short npt) {
    int m = x.n_elem;
    arma::vec J = arma::zeros(m);
    for (int i=0; i < m; ++i) {
        auto ff = [&f,&x,i](double z) -> double {
            arma::vec u = x;
            u(i) = z;
            u = f(u);
            return u(i);
        };
        J(i) = deriv(ff, x(i), h, catch_zero, npt);
    }
    return J;
}