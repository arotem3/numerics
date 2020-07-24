#include "numerics.hpp"

void numerics::CubicInterp::fit(const arma::vec& X, const arma::mat& Y) {
    _check_xy(X, Y);
    _check_x(X);

    u_long n = X.n_elem - 1;

    arma::mat h = arma::zeros(n,1);
    arma::sp_mat A(n+1,n+1);
    arma::mat RHS = arma::zeros(n+1,Y.n_cols);
    _b = arma::zeros(n,Y.n_cols);
    _d = arma::zeros(n,Y.n_cols);
    arma::uvec I = arma::sort_index(X);
    _x = X(I);
    _y = Y.rows(I);
    _lb = _x.front(); _lb -= std::max(1e-8*std::abs(_lb), 1e-10);
    _ub = _x.back(); _ub += std::max(1e-8*std::abs(_ub), 1e-10);

    for (u_long i=1; i < n+1; ++i) {
        h(i-1) = _x(i) - _x(i-1); 
    }

    arma::vec subD = h;
    arma::vec supD(arma::size(subD),arma::fill::zeros);
    arma::vec mainD(n+1,arma::fill::zeros);

    subD(n-1) = 0;
    mainD(0) = 1;
    mainD(n) = 1;
    supD(0) = 0;

    for (u_long i=1; i < n; ++i) {     
        mainD(i) = 2 * (h(i) + h(i-1));
        supD(i) = h(i);

        RHS.row(i) = 3 * (y.row(i+1) - y.row(i))/h(i) - 3 * (y.row(i) - y.row(i-1))/h(i-1);
    }

    A.diag(-1) = subD;
    A.diag()   = mainD;
    A.diag(1)  = supD;

    _c = spsolve(A,RHS);

    for (u_long i=0; i < n; ++i) {
        _b.row(i) = (_y.row(i+1) - _y.row(i))/h(i) - h(i)*(2*_c.row(i) + _c.row(i+1))/3;
        _d.row(i) = (_c.row(i+1) - _c.row(i))/(3*h(i));
    }
    _c = _c.rows(arma::span(0,n-1));
}

void numerics::CubicInterp::save(std::ostream& out) const {
    out << _y.n_rows << " " << _y.n_cols << std::endl;
    out.precision(12);
    _b.t().raw_print(out);
    _c.t().raw_print(out);
    _d.t().raw_print(out);
    _x.t().raw_print(out);
    _y.t().raw_print(out);
}

void numerics::CubicInterp::load(std::istream& in) {
    u_long n, m;
    in >> n >> m;
    _b = arma::zeros(n,m);
    _c = arma::zeros(n,m);
    _d = arma::zeros(n,m);
    _x = arma::zeros(n+1);
    _y = arma::zeros(n+1,m);
    for (u_long i=0; i < m; ++i) {
        for (u_long j=0; j < n; ++j) {
            in >> _b(j,i);
        }
    }
    for (u_long i=0; i < m; ++i) {
        for (u_long j=0; j < n; ++j) {
            in >> _c(j,i);
        }
    }
    for (u_long i=0; i < m; ++i) {
        for (u_long j=0; j < n; ++j) {
            in >> _d(j,i);
        }
    }
    for (u_long i=0; i < n+1; ++i) {
        in >> _x(i);
    }
    for (u_long i=0; i < m; ++i) {
        for (u_long j=0; j < n+1; ++j) {
            in >> _y(j,i);
        }
    }
}

arma::mat numerics::CubicInterp::predict(const arma::vec& t) const {
    _check_range(t);

    u_long t_length = t.n_elem;
    arma::mat s = arma::zeros(t_length,_y.n_cols);

    for (u_long i=0; i < t_length; ++i) {
        for (u_long j=0; j < _x.n_elem-1; ++j) {
            if (_x(j) <= t(i) && t(i) <= _x(j+1)) {
                double h = t(i) - _x(j);
                s.row(i) = _y.row(j) + _b.row(j)*h + _c.row(j)*std::pow(h,2) + _d.row(j)*std::pow(h,3);
            }
        }
    }

    return s;
}