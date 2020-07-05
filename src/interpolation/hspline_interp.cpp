#include <numerics.hpp>

numerics::hspline_interp::hspline_interp() {
    // do nothing
}

numerics::hspline_interp::hspline_interp(std::istream& in) {
    load(in);
}

numerics::hspline_interp::hspline_interp(const arma::vec& X, const arma::mat& Y, const arma::mat& Yp) {
    fit(X,Y,Yp);
}

numerics::hspline_interp::hspline_interp(const arma::vec& X, const arma::mat& Y) {
    fit(X,Y);
}

numerics::hspline_interp& numerics::hspline_interp::fit(const arma::vec& X, const arma::mat& Y) {
    if ( X.n_rows != Y.n_rows ) { // error with input arguments
        std::cerr << "hspline_interp() error: interpolation could not be constructed.\nInput number of elements in x (=" << X.n_elem << ") does not equal the number of rows in y (=" << Y.n_rows << ").\n";
        return *this;
    }
    
    int n = X.n_elem, m = Y.n_cols;
    for (uint i=0; i < n; ++i) {
        if (x.count(X(i)) > 0) {
            std::cerr << "hspline_interp() error: interpolation could not be constructed.\nOne or more repeated x values.\n";
            x.clear();
            return *this;
        }
        else x.insert({X(i),i});
    }
    y = Y;

    arma::sp_mat D;
    ode::diffmat(D,X);
    dy = D*y;

    arma::mat h = X.rows(1,n-1) - X.rows(0,n-2);
    h = arma::repmat(h,1,m);
    a = (2*(y.rows(0,n-2) - y.rows(1,n-1)) + (dy.rows(0,n-2) + dy.rows(1,n-1))%h) / arma::pow(h,3);
    b = -(3*(y.rows(0,n-2) - y.rows(1,n-1)) + (2*dy.rows(0,n-2) + dy.rows(1,n-1))%h) / arma::pow(h,2);
    return *this;
}

numerics::hspline_interp& numerics::hspline_interp::fit(const arma::vec& X, const arma::mat& Y, const arma::mat& Yp) {
    if ( X.n_rows != Y.n_rows ) { // error with input arguments
        std::cerr << "hspline_interp() error: interpolation could not be constructed.\nInput number of elements in x (=" << X.n_elem << ") does not equal the number of rows in y (=" << Y.n_rows << ").\n";
        return *this;
    }
    
    int n = X.n_elem, m = Y.n_cols;
    for (uint i=0; i < n; ++i) {
        if (x.count(X(i)) > 0) {
            std::cerr << "hspline_interp() error: interpolation could not be constructed.\nOne or more repeated x values.\n";
            x.clear();
            return *this;
        }
        else x.insert({X(i),i});
    }
    y = Y;
    dy = Yp;

    arma::mat h = X.rows(1,n-1) - X.rows(0,n-2);
    h = arma::repmat(h,1,m);
    a = (2*(y.rows(0,n-2) - y.rows(1,n-1)) + (dy.rows(0,n-2) + dy.rows(1,n-1))%h) / arma::pow(h,3);
    b = -(3*(y.rows(0,n-2) - y.rows(1,n-1)) + (2*dy.rows(0,n-2) + dy.rows(1,n-1))%h) / arma::pow(h,2);
    return *this;
}

arma::mat numerics::hspline_interp::operator()(const arma::vec& xx) {
    return predict(xx);
}

arma::mat numerics::hspline_interp::predict(const arma::vec& xx) {
    arma::mat yy(xx.n_elem, y.n_cols);
    double minx = x.begin()->first, maxx = x.rbegin()->first;
    minx = minx - 0.01*std::abs(minx);
    maxx = maxx + 0.01*std::abs(maxx);
    for (uint i=0; i < xx.n_elem; ++i) {
        if (xx(i) < minx || maxx < xx(i)) {
            std::cerr << "hspline::predict() error: one or more values are outside the range of interpolation: [" << minx << " , " << maxx << "]\n"
                      << "returning zero matrix...\n";
            return arma::zeros(xx.n_elem,y.n_cols);
        }
        auto x0 = x.lower_bound(xx(i));
        if (x0->first > xx(i) || x0->second >= y.n_rows-1) x0--;
        double h = xx(i) - x0->first;
        int j = x0->second;
        yy.row(i) = a.row(j)*(h*h*h) + b.row(j)*h*h + dy.row(j)*h + y.row(j);
    }
    return yy;
}

arma::mat numerics::hspline_interp::predict_derivative(const arma::vec& xx) {
    arma::mat yy(xx.n_elem, y.n_cols);
    double minx = x.begin()->first, maxx = x.rbegin()->first;
    minx = minx - 0.01*std::abs(minx);
    maxx = maxx + 0.01*std::abs(maxx);
    for (uint i=0; i < xx.n_elem; ++i) {
        if (xx(i) < minx || maxx < xx(i)) {
            std::cerr << "hspline::predict_derivative() error: one or more values are outside the range of interpolation: [" << minx << " , " << maxx << "]\n"
                      << "returning zero matrix...\n";
            return arma::zeros(xx.n_elem,y.n_cols);
        }
        auto x0 = x.lower_bound(xx(i));
        if (x0->first > xx(i) || x0->second >= y.n_rows-1) x0--;
        double h = xx(i) - x0->first;
        int j = x0->second;
        yy.row(i) = 3*a.row(j)*h*h + 2*b.row(j)*h + dy.row(j);
    }
    return yy;
}

arma::vec numerics::hspline_interp::data_X() {
    arma::vec X(y.n_rows);
    for (auto xx : x) {
        X(xx.second) = xx.first;
    }
    return X;
}

arma::mat numerics::hspline_interp::data_Y() {
    return y;
}

arma::mat numerics::hspline_interp::data_dY() {
    return dy;
}

void numerics::hspline_interp::save(std::ostream& out) {
    out.precision(12);
    out << y.n_rows << " " << y.n_cols << std::endl;
    a.t().raw_print(out);
    b.t().raw_print(out);
    data_X().t().raw_print(out);
    y.t().raw_print(out);
    dy.t().raw_print(out);
}

void numerics::hspline_interp::load(std::istream& in) {
    int m,n;
    in >> n >> m;

    a.set_size(n-1,m);
    b.set_size(n-1,m);
    y.set_size(n,m);
    dy.set_size(n,m);

    for (int i(0); i < m; ++i) {
        for (int j(0); j < n-1; ++j) {
            in >> a(j,i);
        }
    }
    for (int i(0); i < m; ++i) {
        for (int j(0); j < n-1; ++j) {
            in >> b(j,i);
        }
    }
    double xx;
    for (int i(0); i < n; ++i) {
        in >> xx;
        x[xx] = i;
    }
    for (int i(0); i < m; ++i) {
        for (int j(0); j < n; ++j) {
            in >> y(j,i);
        }
    }
    for (int i(0); i < m; ++i) {
        for (int j(0); j < n; ++j) {
            in >> dy(j,i);
        }
    }
}