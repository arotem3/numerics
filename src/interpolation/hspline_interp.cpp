#include <numerics.hpp>

numerics::PieceWisePoly numerics::hermite_cubic_spline(const arma::vec& x, const arma::mat& y, const std::string& extrapolation, double val) {
    u_long dim = y.n_cols;
    PieceWisePoly out(extrapolation, val, dim);
    
    if (x.n_elem != y.n_rows) {
        throw std::invalid_argument("hermite_cubic_spline() error: dimension mismatch, x.n_elem (=" + std::to_string(x.n_elem) + ") != y.n_rows (=" + std::to_string(y.n_elem) + ").");
    }

    u_long n = x.n_elem;
    arma::uvec I = arma::sort_index(x);
    arma::vec _x = x(I);
    for (u_long i=0; i < n-1; ++i) {
        if (_x(i+1) - _x(i) < std::abs(_x(i+1))*arma::datum::eps) throw std::runtime_error("hermite_cubic_spline() error: atleast two elements in x are within epsilon of each other.");
    }
    arma::mat _y = y.rows(I);

    arma::sp_mat D;
    ode::diffmat(D, _x);
    arma::mat _dy = D*_y;

    for (u_long i=0; i < n-1; ++i) {
        arma::mat p(4, dim);
        double h = _x(i+1) - _x(i);
        p.row(0) = 0.125 * (h*_dy.row(i) + h*_dy.row(i+1) +   2*_y.row(i) - 2*_y.row(i+1));
        p.row(1) = 0.125 * h * (-_dy.row(i) + _dy.row(i+1));
        p.row(2) = 0.125 * (-h*_dy.row(i) - h*_dy.row(i+1) - 6*_y.row(i) + 6*_y.row(i+1));
        p.row(3) = 0.125 * (h*_dy.row(i) - h*_dy.row(i+1) + 4*_y.row(i) + 4*_y.row(i+1));
        out.push(Polynomial(std::move(p), _x(i), _x(i+1)));
    }
    return out;
}

numerics::PieceWisePoly numerics::hermite_cubic_spline(const arma::vec& x, const arma::mat& y, const arma::mat& yp, const std::string& extrapolation, double val) {
    u_long dim = y.n_cols;
    PieceWisePoly out(extrapolation, val, dim);
    
    if (x.n_elem != y.n_rows) {
        throw std::invalid_argument("hermite_cubic_spline() error: dimension mismatch, x.n_elem (=" + std::to_string(x.n_elem) + ") != y.n_rows (=" + std::to_string(y.n_elem) + ").");
    }

    u_long n = x.n_elem;
    arma::uvec I = arma::sort_index(x);
    arma::vec _x = x(I);
    for (u_long i=0; i < n; ++i) {
        if (_x(i+1) - _x(i) < std::abs(_x(i+1))*arma::datum::eps) throw std::runtime_error("hermite_cubic_spline() error: atleast two elements in x are within epsilon of each other.");
    }
    arma::mat _y = y.rows(I);
    arma::mat _dy = yp.rows(I);

    for (u_long i=0; i < n-1; ++i) {
        arma::mat p(4, dim);
        p.row(0) = 0.25 * ( _dy.row(i) + _dy.row(i+1) +   _y.row(i) -   _y.row(i+1));
        p.row(1) = 0.25 * (-_dy.row(i) + _dy.row(i+1));
        p.row(2) = 0.25 * (-_dy.row(i) - _dy.row(i+1) - 3*_y.row(i) + 3*_y.row(i+1));
        p.row(3) = 0.25 * ( _dy.row(i) - _dy.row(i+1) + 2*_y.row(i) + 2*_y.row(i+1));
        out.push(Polynomial(std::move(p), _x(i), _x(i+1)));
    }
    return out;
}