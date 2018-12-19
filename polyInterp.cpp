#include "numerics.hpp"

numerics::polyInterp::polyInterp() {
    // DOES NOTHING
}


numerics::polyInterp::polyInterp(const arma::vec& X, const arma::mat& Y) {
    if ( X.n_rows != Y.n_rows ) { // error with input arguments
        std::cerr << "polyInterp() error: interpolation could not be constructed. Input vectors must have the same length." << std::endl;
        x = {0};
        p = {0};
        return;
    }
    
    int n = X.n_elem - 1;
    
    for (int i(0); i < n; ++i) { // error with invalid x input
        for (int j(i+1); j < n+1; ++j) {
            if ( std::abs(X(i) - X(j)) < eps(X(i)) ) {
                std::cerr << "polyInterp() error: one or more x values are repeting, therefore no polynomial interpolation exists for this data." << std::endl;
                x = {0};
                p = {0};
                return;
            }
        }
    }

    x = X;
    p = arma::zeros(n+1,Y.n_cols);
    for (int i(0); i < Y.n_cols; ++i) {
        p.col(i) = arma::polyfit(X,Y.col(i),n);
    }
}

numerics::polyInterp::polyInterp(std::istream& in) {
    load(in);
}

void numerics::polyInterp::load(std::istream& in) {
    int n, m;
    in >> n >> m;
    x = arma::zeros(n);
    p = arma::zeros(n,m);
    for (int i(0); i < n; ++i) {
        in >> x(i);
    }
    for (int i(0); i < m; ++i) {
        for (int j(0); j < n; ++j) {
            in >> p(j,i);
        }
    }
}

void numerics::polyInterp::save(std::ostream& out) {
    out << p.n_rows << " " << p.n_cols << std::endl;
    out.precision(12);
    arma::mat temp = x.t();
    temp.raw_print(out);
    temp = p.t();
    temp.raw_print(out);
}

arma::mat numerics::polyInterp::operator()(const arma::vec& u) {
    int n = x.n_elem - 1;
    if ( !arma::all(u - x(0) >= -0.01) || !arma::all(u - x(n) <= 0.01) ) { // input error
        std::cerr << "polyInterp::operator() failed: one or more input value is outside the domain of the interpolation. No possible evaluation exists." << std::endl;
        return {NAN};
    }
    arma::mat v(u.n_elem,p.n_cols, arma::fill::zeros);
    for (int i(0); i < p.n_cols; ++i) {
        v.col(i) = arma::polyval(p.col(i),u);
    }
    return v;
}