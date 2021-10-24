#include "numerics.hpp"

numerics::PieceWisePoly numerics::natural_cubic_spline(const arma::vec& x, const arma::mat&y, const std::string& extrapolation, double val) {
    u_long dim = y.n_cols;
    PieceWisePoly out(extrapolation, val, dim);
    
    if (x.n_elem != y.n_rows) {
        throw std::invalid_argument("natural_cubic_spline() error: dimension mismatch, x.n_elem (=" + std::to_string(x.n_elem) + ") != y.n_rows (=" + std::to_string(y.n_elem) + ").");
    }

    u_long n = x.n_elem - 1;

    arma::sp_mat A(n+1,n+1);
    arma::mat RHS = arma::zeros(n+1, dim);
    arma::mat b = arma::zeros(n, dim);
    arma::mat d = arma::zeros(n, dim);
    arma::uvec I = arma::sort_index(x);
    arma::vec _x = x(I);
    for (u_long i=0; i < n; ++i) {
        if (_x(i+1) - _x(i) < std::abs(_x(i+1))*arma::datum::eps) throw std::runtime_error("natural_cubic_spline() error: atleast two elements in x are within epsilon of each other.");
    }
    arma::mat _y = y.rows(I);

    arma::vec h = arma::diff(_x);

    arma::vec subD = h;
    arma::vec supD = arma::zeros(n);
    arma::vec mainD = arma::zeros(n+1);

    subD(n-1) = 0;
    mainD(0) = 1;
    mainD(n) = 1;
    supD(0) = 0;

    for (u_long i=1; i < n; ++i) {     
        mainD(i) = 2 * (h(i) + h(i-1));
        supD(i) = h(i);

        RHS.row(i) = 3 * (_y.row(i+1) - _y.row(i))/h(i) - 3 * (_y.row(i) - _y.row(i-1))/h(i-1);
    }

    A.diag(-1) = subD;
    A.diag()   = mainD;
    A.diag(1)  = supD;

    arma::mat c = spsolve(A,RHS);

    for (u_long i=0; i < n; ++i) {
        b.row(i) = (_y.row(i+1) - _y.row(i))/h(i) - h(i)*(2*c.row(i) + c.row(i+1))/3;
        d.row(i) = (c.row(i+1) - c.row(i))/(3*h(i));
    }
    c = c.rows(0,n-1);

    for (u_long i=0; i < n; ++i) {
        arma::mat p(4, dim);
        // translate from p(x - x[i]) to  p(z) for z = 2*(x - x[i])/(x[i+1] - x[i]) - 1
        p.row(0) = 0.125 * std::pow(h(i),3) * d.row(i); // h^3 d / 8
        p.row(1) = 0.125 * std::pow(h(i),2) * (2*c.row(i) + 3*h(i)*d.row(i)); // (2c + 3h*d)* h^2 / 8
        p.row(2) = 0.125 * h(i) * (4*b.row(i) + 4*h(i)*c.row(i) + 3*std::pow(h(i),2)*d.row(i)); // (4b +4h*c + 3*h^2*d) * h / 8
        p.row(3) = _y.row(i) + 0.125 * h(i) * (4*b.row(i) + 2*h(i)*c.row(i) + std::pow(h(i),2)*d.row(i)); // y + (4b + 2hc + h^2*d) * h / 8
        out.push(Polynomial(std::move(p), _x(i), _x(i+1)));
    }
    return out;
}