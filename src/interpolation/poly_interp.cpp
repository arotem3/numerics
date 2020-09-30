#include "numerics.hpp"

void numerics::PolyInterp::fit(const arma::vec& X, const arma::mat& Y) {
    _check_xy(X,Y);
    _check_x(X);

    _p.clear();
    for (int i=0; i < Y.n_cols; ++i) {
        _p.push_back(Polynomial(X, Y.col(i)));
    }
}

void numerics::PolyInterp::load(std::istream& in) {
    u_long n, m;
    in >> n >> m >> _extrap >> _extrap_val >> _lb >> _ub;
    arma::mat p = arma::zeros(n,m);
    for (u_long i(0); i < m; ++i) {
        for (u_long j(0); j < n; ++j) {
            in >> p(j,i);
        }
    }
    _p.clear();
    for (int i=0; i < m; ++i) {
        _p.push_back(Polynomial(p.col(i)));
    }
}

/* save(out) : save data structure to file.
 * --- out : file/output stream pointing to write data to. */
void numerics::PolyInterp::save(std::ostream& out) const {
    out << _p.n_rows << " " << _p.n_cols << " " << _extrap << " " << _extrap_val << " " << _lb << " " << _ub << std::endl;
    out.precision(12);
    for (const Polynomial& P : _p) {
        P.coefficients.t().raw_print(out);
    }
}

/* predict(t) : evaluate interpolator like a function at specific values.
 * --- t : points to evaluate interpolation on. */
arma::mat numerics::PolyInterp::predict(const arma::vec& u) const {
    arma::mat v = arma::zeros(u.n_elem, _dim);
    if (_extrap == 0) { // const
        v += _extrap_val;
        arma::uvec I = arma::find((_lb <= u) and (u <= _ub));
        arma::mat vin(I.n_elem, _dim);
        for (u_int i=0; i < _dim; ++i) {
            vin.col(i) = _p.at(i)(u);
        }
        v.rows(I) = vin;
    } else if (_extrap == 1) { // boundary
        arma::vec t = _flat_past_boundary(u);
        for (u_int i=0; i < _dim; ++i) {
            v.col(i) = _p.at(i)(u);
        }
    } else if (_extrap == 2) { // linear
        arma::uvec I = arma::find((_lb <= u) and (u <= _ub));
        arma::mat vin(I.n_elem, _dim);
        for (u_int i=0; i < _dim; ++i) {
            vin.col(i) = _p.at(i)(u);
        }
        v.rows(I) = vin;

        I = arma::find(u < _lb);
        arma::mat lv(I.n_elem, _dim);
        for (u_int i=0; i < _dim; ++i) {
            double p0 = _p.at(i)(_lb);
            Polynomial dp = _p.at(i).derivative();
            double p1 = dp(_lb);
            lv.col(i) = p0 + p1*(u(I) - _lb);
        }
        v.rows(I) = lv;

        I = arma::find(u > _ub);
        arma::mat uv(I.n_elem, _dim);
        for (u_int i=0; i < _dim; ++i) {
            double p0 = _p.at(i)(_ub);
            Polynomial dp = _p.at(i).derivative();
            double p1 = dp(_ub);
            lv.col(i) = p0 + p1*(u(I) - _ub);
        }
        v.rows(I) = uv;
    } else if (_extrap == 3) { //periodic
        arma::vec t = _periodic(u);
        for (u_int i=0; i < _dim; ++i) {
            v.col(i) = _p.at(i)(u);
        }
    } else { // polynomial
        for (u_int i=0; i < _dim; ++i) {
            v.col(i) = _p.at(i)(u);
        }
    }
    return v;
}