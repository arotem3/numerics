#include <numerics.hpp>

void numerics::HSplineInterp::fit(const arma::vec& X, const arma::mat& Y) {
    _check_xy(X,Y);
    
    u_long n = X.n_elem, m = Y.n_cols;
    _check_x(X);
    arma::uvec I = arma::sort_index(X);
    _x = X(I);
    _y = Y.rows(I);
    _lb = _x.front(); _lb -= std::max(1e-8*std::abs(_lb), 1e-10);
    _ub = _x.back(); _ub += std::max(1e-8*std::abs(_ub), 1e-10);

    arma::sp_mat D;
    ode::diffmat(D,_x);
    _dy = D*_y;

    arma::mat h = _x.rows(1,n-1) - _x.rows(0,n-2);
    h = arma::repmat(h,1,m);
    _a = (2*(_y.rows(0,n-2) - _y.rows(1,n-1)) + (_dy.rows(0,n-2) + _dy.rows(1,n-1))%h) / arma::pow(h,3);
    _b = -(3*(_y.rows(0,n-2) - _y.rows(1,n-1)) + (2*_dy.rows(0,n-2) + _dy.rows(1,n-1))%h) / arma::pow(h,2);
}

void numerics::HSplineInterp::fit(const arma::vec& X, const arma::mat& Y, const arma::mat& Yp) {
    _check_xy(X,Y);
    _check_xy(X,Yp);
    if (Y.n_cols != Yp.n_cols) {
        throw std::invalid_argument("dimension mismatch, y.n_cols (=" + std::to_string(Y.n_cols) + ") != yp.n_cols (=" + std::to_string(Yp.n_cols) + ")");
    }
    
    u_long n = X.n_elem, m = Y.n_cols;
    _check_x(X);
    arma::uvec I = arma::sort_index(X);
    _x = X(I);
    _y = Y.rows(I);
    _dy = Yp.rows(I);
    _lb = _x.front(); _lb -= std::max(1e-8*std::abs(_lb), 1e-10);
    _ub = _x.back(); _ub += std::max(1e-8*std::abs(_ub), 1e-10);

    arma::mat h = _x.rows(1,n-1) - _x.rows(0,n-2);
    h = arma::repmat(h,1,m);
    _a = (2*(_y.rows(0,n-2) - _y.rows(1,n-1)) + (_dy.rows(0,n-2) + _dy.rows(1,n-1))%h) / arma::pow(h,3);
    _b = -(3*(_y.rows(0,n-2) - _y.rows(1,n-1)) + (2*_dy.rows(0,n-2) + _dy.rows(1,n-1))%h) / arma::pow(h,2);
}

arma::mat numerics::HSplineInterp::predict(const arma::vec& xx) const {
    arma::mat yy(xx.n_elem, y.n_cols);
    _check_range(xx);
    
    for (u_long i=0; i < xx.n_elem; ++i) {
        for (u_long j=0; j < _x.n_elem-1; ++j) {
            if (_x(j) <= xx(i) && xx(i) <= _x(j+1)) {
                double h = xx(i) - _x(j);
                yy.row(i) = _a.row(j)*std::pow(h,3) + _b.row(j)*std::pow(h,2) + _dy.row(j)*h + _y.row(j);
            }
        }
    }

    return yy;
}

arma::mat numerics::HSplineInterp::predict_derivative(const arma::vec& xx) const {
    arma::mat yy(xx.n_elem, y.n_cols);
    _check_range(xx);

    for (u_long i=0; i < xx.n_elem; ++i) {
        for (u_long j=0; j < _x.n_elem-1; ++j) {
            if (_x(j) <= xx(i) && xx(i) <= _x(j+1)) {
                double h = xx(i) - _x(j);
                yy.row(i) = 3*_a.row(j)*std::pow(h,2) + 2*_b.row(j)*h + _dy.row(j);
            }
        }
    }

    return yy;
}

void numerics::HSplineInterp::save(std::ostream& out) const {
    out.precision(12);
    out << _y.n_rows << " " << _y.n_cols << std::endl;
    _a.t().raw_print(out);
    _b.t().raw_print(out);
    _x.t().raw_print(out);
    _y.t().raw_print(out);
    _dy.t().raw_print(out);
}

void numerics::HSplineInterp::load(std::istream& in) {
    u_long m,n;
    in >> n >> m;

    _a.set_size(n-1,m);
    _b.set_size(n-1,m);
    _y.set_size(n,m);
    _dy.set_size(n,m);

    for (u_long i=0; i < m; ++i) {
        for (u_long j=0; j < n-1; ++j) {
            in >> _a(j,i);
        }
    }
    for (u_long i=0; i < m; ++i) {
        for (u_long j=0; j < n-1; ++j) {
            in >> _b(j,i);
        }
    }
    for (u_long i=0; i < n; ++i) {
        in >> _x(i);
    }
    for (u_long i=0; i < m; ++i) {
        for (u_long j=0; j < n; ++j) {
            in >> _y(j,i);
        }
    }
    for (u_long i=0; i < m; ++i) {
        for (u_long j=0; j < n; ++j) {
            in >> _dy(j,i);
        }
    }
}