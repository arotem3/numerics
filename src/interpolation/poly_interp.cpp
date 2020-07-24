#include "numerics.hpp"

void numerics::PolyInterp::fit(const arma::vec& X, const arma::mat& Y) {
    _check_xy(X,Y);
    
    u_long n = X.n_elem - 1;
    
    _check_x(X);

    _lb = X.min(); _lb -= 1e-8*std::abs(_lb);
    _ub = X.max(); _ub -= 1e-8*std::abs(_ub);

    _p = arma::zeros(n+1,Y.n_cols);
    for (u_long i(0); i < Y.n_cols; ++i) {
        _p.col(i) = arma::polyfit(X,Y.col(i),n);
    }
}

void numerics::PolyInterp::load(std::istream& in) {
    u_long n, m;
    in >> n >> m;
    _p = arma::zeros(n,m);
    for (u_long i(0); i < m; ++i) {
        for (u_long j(0); j < n; ++j) {
            in >> _p(j,i);
        }
    }
}

/* save(out) : save data structure to file.
 * --- out : file/output stream pointing to write data to. */
void numerics::PolyInterp::save(std::ostream& out) const {
    out << _p.n_rows << " " << _p.n_cols << std::endl;
    out.precision(12);
    _p.t().raw_print(out);
}

/* predict(t) : evaluate interpolator like a function at specific values.
 * --- t : points to evaluate interpolation on. */
arma::mat numerics::PolyInterp::predict(const arma::vec& u) const {
    arma::mat v(u.n_elem,_p.n_cols, arma::fill::zeros);
    for (u_long i(0); i < _p.n_cols; ++i) {
        v.col(i) = arma::polyval(_p.col(i),u);
    }
    return v;
}