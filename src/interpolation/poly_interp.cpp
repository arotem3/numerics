#include "numerics.hpp"

/* poly_interp() : empty constructor does nothing */
numerics::poly_interp::poly_interp() {
    // DOES NOTHING
}

/* poly_interp(x, y) : build polynomial interpolator
 * --- X : domain
 * --- Y : values to interpolate, each col is a different set of values */
numerics::poly_interp::poly_interp(const arma::vec& X, const arma::mat& Y) {
    if ( X.n_rows != Y.n_rows ) { // error with input arguments
        std::cerr << "poly_interp() error: interpolation could not be constructed. Input vectors must have the same length." << std::endl;
        x = {0};
        p = {0};
        return;
    }
    
    int n = X.n_elem - 1;
    
    for (int i(0); i < n; ++i) { // error with invalid x input
        for (int j(i+1); j < n+1; ++j) {
            if ( std::abs(X(i) - X(j)) < arma::eps(X(i)) ) {
                std::cerr << "poly_interp() error: one or more x values are repeting, therefore no polynomial interpolation exists for this data." << std::endl;
                x = {0};
                p = {0};
                return;
            }
        }
    }

    x = X;
    y = Y;
    p = arma::zeros(n+1,Y.n_cols);
    for (int i(0); i < Y.n_cols; ++i) {
        p.col(i) = arma::polyfit(X,Y.col(i),n);
    }
}

/* poly_interp(in) : load data structure from file
 * --- in : file/input stream pointing to top of cubic interpolator object */
numerics::poly_interp::poly_interp(std::istream& in) {
    load(in);
}

/* fit(x, y) : fit the object, same as initialization */
numerics::poly_interp& numerics::poly_interp::fit(const arma::vec& X, const arma::mat& Y) {
    poly_interp(X,Y);
    return *this;
}

/* load(in) : load data structure from file
 * --- in : file/input stream pointing to top of cubic interpolator object */
void numerics::poly_interp::load(std::istream& in) {
    int n, m;
    in >> n >> m;
    x = arma::zeros(n);
    y = arma::zeros(n,m);
    p = arma::zeros(n,m);
    for (int i(0); i < n; ++i) {
        in >> x(i);
    }
    for (int i(0); i < m; ++i) {
        for (int j(0); j < n; ++j) {
            in >> y(j,i);
        }
    }
    for (int i(0); i < m; ++i) {
        for (int j(0); j < n; ++j) {
            in >> p(j,i);
        }
    }
}

/* save(out) : save data structure to file.
 * --- out : file/output stream pointing to write data to. */
void numerics::poly_interp::save(std::ostream& out) {
    out << p.n_rows << " " << p.n_cols << std::endl;
    out.precision(12);
    x.t().raw_print(out);
    y.t().raw_print(out);
    p.t().raw_print(out);
}

/* poly_interp::(t) : same as predict(t) */
arma::mat numerics::poly_interp::operator()(const arma::vec& u) {
    return predict(u);
}

/* predict(t) : evaluate interpolator like a function at specific values.
 * --- t : points to evaluate interpolation on. */
arma::mat numerics::poly_interp::predict(const arma::vec& u) {
    arma::mat v(u.n_elem,p.n_cols, arma::fill::zeros);
    for (int i(0); i < p.n_cols; ++i) {
        v.col(i) = arma::polyval(p.col(i),u);
    }
    return v;
}


arma::vec numerics::poly_interp::data_X() {
    return x;
}

arma::mat numerics::poly_interp::data_Y() {
    return y;
}

arma::mat numerics::poly_interp::coefficients() const {
    return p;
}