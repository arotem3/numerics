#include "numerics.hpp"

/* CUBICINTERP : empty constructor does nothing */
numerics::CubicInterp::CubicInterp() {
    n = 0;
}

/* CUBICINTERP : cubic interpolation with one independent variable
 * --- x : independent variable
 * --- y : dependent variable */
numerics::CubicInterp::CubicInterp(const arma::vec& X, const arma::mat& Y) {
    if ( X.n_rows != Y.n_rows ) { // error with input arguments
        std::cerr << "CubicInterp() error: interpolation could not be constructed. Input vectors must have the same length." << std::endl;
        n = -1;
        b = {NAN};
        c = {NAN};
        d = {NAN};
        x = {NAN};
        y = {NAN};
        return;
    }
    
    n = X.n_elem - 1;
    
    for (int i(0); i < n; ++i) { // error with invalid x input
        for (int j(i+1); j < n+1; ++j) {
            if ( std::abs(X(i) - X(j)) < eps(X(i)) ) {
                std::cerr << "CubicInterp() error: one or more x values are repeting, therefore no cubic interpolation exists for this data." << std::endl;
                n = -1;
                b = {NAN};
                c = {NAN};
                d = {NAN};
                x = {NAN};
                y = {NAN};
                return;
            }
        }
    }

    arma::mat h = arma::zeros(n,1);
    arma::sp_mat A(n+1,n+1);
    arma::mat RHS = arma::zeros(n+1,Y.n_cols);
    b = arma::zeros(n,Y.n_cols);
    d = arma::zeros(n,Y.n_cols);
    x = X;
    y = Y;

    for (int i(1); i < n+1; ++i) {
        h(i-1) = x(i) - x(i-1); 
    }

    arma::vec subD = h;
    arma::vec supD(arma::size(subD),arma::fill::zeros);
    arma::vec mainD(n+1,arma::fill::zeros);

    subD(n-1) = 0;
    mainD(0) = 1;
    mainD(n) = 1;
    supD(0) = 0;

    for (int i(1); i < n; ++i) {     
        mainD(i) = 2 * (h(i) + h(i-1));
        supD(i) = h(i);

        RHS.row(i) = 3 * (y.row(i+1) - y.row(i))/h(i) - 3 * (y.row(i) - y.row(i-1))/h(i-1);
    }

    A.diag(-1) = subD;
    A.diag()   = mainD;
    A.diag(1)  = supD;

    c = spsolve(A,RHS);

    for (int i(0); i < n; ++i) {
        b.row(i) = (y.row(i+1) - y.row(i))/h(i) - h(i)*(2*c.row(i) + c.row(i+1))/3;
        d.row(i) = (c.row(i+1) - c.row(i))/(3*h(i));
    }
    c = c.rows(arma::span(0,n-1));
}

/* CUBICINTERP : load data structure from file
 * --- in : file/input stream pointing to top of cubic interpolator object */
numerics::CubicInterp::CubicInterp(std::istream& in) {
    load(in);
}

/* SAVE : save data structure to file.
 * --- out : file/output stream pointing to write data to. */
void numerics::CubicInterp::save(std::ostream& out) {
    out << n << " " << y.n_cols << std::endl;
    out.precision(12);
    arma::mat temp = b.t();
    temp.raw_print(out);
    temp = c.t();
    temp.raw_print(out);
    temp = d.t();
    temp.raw_print(out);
    temp = x.t();
    temp.raw_print(out);
    temp = y.t();
    temp.raw_print(out);
}

/* LOAD : load data structure from file
 * --- in : file/input stream pointing to top of cubic interpolator object */
void numerics::CubicInterp::load(std::istream& in) {
    int m;
    in >> n >> m;
    b = arma::zeros(n,m);
        c = arma::zeros(n,m);
        d = arma::zeros(n,m);
        x = arma::zeros(n+1);
        y = arma::zeros(n+1,m);
    for (int i(0); i < m; ++i) {
        for (int j(0); j < n; ++j) {
            in >> b(j,i);
        }
    }
    for (int i(0); i < m; ++i) {
        for (int j(0); j < n; ++j) {
            in >> c(j,i);
        }
    }
    for (int i(0); i < m; ++i) {
        for (int j(0); j < n; ++j) {
            in >> d(j,i);
        }
    }
    for (int i(0); i < n+1; ++i) {
        in >> x(i);
    }
    for (int i(0); i < m; ++i) {
        for (int j(0); j < n+1; ++j) {
            in >> y(j,i);
        }
    }  
}

/* CUBICINTERP({x1,x2,...}) : evaluate interpolator like a function at specific values.
 * --- t : points to evaluate interpolation on. */
arma::mat numerics::CubicInterp::operator()(const arma::vec& t) {
    if ( !arma::all(t - x(0) >= -0.01) || !arma::all(t - x(n) <= 0.01) ) { // input error
        std::cerr << "CubicInterp::operator() failed: one or more input value is outside the domain of the interpolation. No possible evaluation exists." << std::endl;
        return {NAN};
    }

    int t_length = arma::size(t)(0);
    arma::mat s = arma::zeros(t_length,y.n_cols);

    for (int i(0); i < t_length; ++i) {
        for (int j(0); j < n; ++j) {
            if (t(i) >= x(j) && t(i) <= x(j+1)) {
                s.row(i) = y.row(j) + b.row(j)*(t(i) - x(j)) + c.row(j)*pow(t(i) - x(j),2) + d.row(j)*pow(t(i)-x(j),3);
            }
        }
    }

    return s;
}

/* CUBICINTERP(n) : evaluate interpolator like a function at uniformly spaced values.
 * --- n : number of points to evaluate interpolation on. */
arma::mat numerics::CubicInterp::operator()(size_t num_pts) {
    arma::vec t = arma::linspace(x(0),x(n),num_pts);
    return operator()(t);
}