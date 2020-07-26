#include <numerics.hpp>

/* soft_threshold(u,v) : __utility for coordinate_lasso__ compute the soft threshold of u parameterized by v. i.e. sign(u) * positive_part(|u| - v) */
inline arma::mat soft_threshold(const arma::mat& u, double v) {
    return arma::sign(u)%(arma::abs(u)-v > 0)%(arma::abs(u)-v);
}

/* compute_rho(y, X, w, i, p) : __utility for coordinate_lasso__ partial computation of the coordinate descent update. */
arma::rowvec compute_rho(const arma::mat& X, const arma::mat& y, const arma::mat& w, u_long i, u_long p) {
    arma::mat dy = y;
    for (u_long j=0; j < p; ++j) {
        if (j==i) continue;
        else dy -= X.col(j) * w.row(j);
    }
    return X.col(i).t()* dy;
}

arma::rowvec compute_rho(const arma::mat& X, const arma::mat& y, const arma::rowvec& b, const arma::mat& w, u_long i, u_long p) {
    arma::mat dy = y;
    for (u_long j=0; j < p; ++j) {
        if (j==i) continue;
        else dy -= X.col(j) * w.row(j);
    }
    dy.each_row() -= b;
    return X.col(i).t() * dy;
}

arma::rowvec compute_rho_intercept(const arma::mat& X, const arma::mat& y, const arma::mat& w) {
    return arma::sum(y - X*w, 0);
}

double loss(const arma::mat& X, const arma::mat& y, const arma::mat& w) {
    arma::mat dy = y - X*w;
    return arma::dot(dy, dy/X.n_rows);
}

double loss(const arma::mat& X, const arma::mat& y, const arma::rowvec& b, const arma::mat& w) {
    arma::mat dy = y - X*w;
    dy.each_row() -= b;
    return arma::dot(dy, dy/X.n_rows);
}

int coordinate_lasso_unsafe(arma::mat& w, const arma::mat& X, const arma::mat& y, double lambda, double tol, u_long max_iter, bool verbose) {
    u_long p = X.n_cols;
    arma::rowvec z = arma::sum(arma::square(X),0);

    numerics::optimization::VerboseTracker T(max_iter);
    if (verbose) T.header();

    u_long n = 0;
    double loss0 = std::numeric_limits<double>::infinity();
    double loss1 = loss(X,y,w);
    while (true) {
        if (std::abs(loss1 - loss0) < tol) {
            if (verbose) T.success_flag();
            return 0;
        }
        if (n >= max_iter) {
            if (verbose) T.max_iter_flag();
            return 1;
        }

        loss0 = loss1;
        for (u_long i=0; i < p; ++i) {
            arma::rowvec rho = compute_rho(X,y,w,i,p);
            w.row(i) = soft_threshold(rho, lambda/2) / z(i);
        }
        loss1 = loss(X,y,w);
        if (verbose) T.iter(n, loss1);
        n++;
    }
}

int coordinate_lasso_unsafe(arma::rowvec& b, arma::mat& w, const arma::mat& X, const arma::mat& y, double lambda, double tol, u_long max_iter, bool verbose) {
    u_long p = X.n_cols;
    arma::rowvec z = arma::sum(arma::square(X),0);    

    numerics::optimization::VerboseTracker T(max_iter);
    if (verbose) T.header();

    u_long n = 0;
    double loss0 = std::numeric_limits<double>::infinity();
    double loss1 = loss(X,y,b,w);
    while (true) {
        if (std::abs(loss1 - loss0) < tol) {
            if (verbose) T.success_flag();
            return 0;
        }
        if (n >= max_iter) {
            if (verbose) T.max_iter_flag();
            return 1;
        }

        loss0 = loss1;
        for (u_long i=0; i <= p; ++i) {
            n++;
            if (i == p) b = compute_rho_intercept(X, y, w) / X.n_elem;
            else {
                arma::rowvec rho = compute_rho(X,y,b,w,i,p);
                w.row(i) = soft_threshold(rho, lambda/2) / z(i);
            }
        }
        loss1 = loss(X,y,b,w);
        if (verbose) T.iter(n, loss1);
        n++;
    }
}

void check_xy(const arma::mat& X, const arma::mat& y, double tol) {
    if (X.n_rows != y.n_rows) throw std::runtime_error("coordinate_lasso(): dimension mismatch, x.n_rows (=" + std::to_string(X.n_rows) + ") != y.n_rows (=" + std::to_string(y.n_rows) + ")");
    if (X.has_nan() or X.has_inf()) throw std::runtime_error("coordinate_lasso(): one or more values of x is NaN or Inf");
    if (y.has_nan() or y.has_inf()) throw std::runtime_error("coordinate_lasso(): one or more values of y is NaN or Inf");

    arma::rowvec z = arma::sum(arma::square(X),0);
    if (arma::any(z < tol/2)) throw std::runtime_error("coordinate_lasso(): one or more columns of X is a zero vector (or nearly zero).");
    if (z.has_nan() or z.has_inf()) throw std::overflow_error("coordinate_lasso(): one or more columns of X has an unbounded norm. Try rescaling.");
}

void check_params(double lambda, double tol, long max_iter) {
    if (lambda < 0) throw std::runtime_error("coordinate_lasso(): require lambda (=" + std::to_string(lambda) + ") > 0");
    if (max_iter < 1) throw std::runtime_error("coordinate_lasso(): require max_iter (=" + std::to_string(max_iter) + ") >= 1");
    if (tol < 0) throw std::runtime_error("coordinate_lasso(): require tol (=" + std::to_string(tol) + ") >= 0");
}

void check_w(arma::mat& w, const arma::mat& X, const arma::mat& y) {
    if ((w.n_rows != X.n_cols) or (w.n_cols != y.n_cols)) w = arma::zeros(X.n_cols, y.n_cols);
    else if (w.has_nan() or w.has_inf()) throw std::runtime_error("coordinate_lasso(): initial coefficients have nan or inf. Try providing an empty array, or an array of zeros.");
}

int numerics::coordinate_lasso(arma::mat& w, const arma::mat& X, const arma::mat& y, double lambda, double tol, u_long max_iter, bool verbose) {
    check_params(lambda, tol, max_iter);
    check_xy(X, y, tol);
    check_w(w, X, y);
    return coordinate_lasso_unsafe(w, X, y, lambda, tol, max_iter, verbose);
}

int numerics::coordinate_lasso(arma::rowvec& b, arma::mat& w, const arma::mat& X, const arma::mat& y, double lambda, double tol, long max_iter, bool verbose) {
    check_params(lambda, tol, max_iter);
    check_xy(X, y, tol);

    if (b.n_cols != y.n_cols) {
        b = arma::mean(y,0);
        if (b.has_nan() or b.has_inf()) throw std::overflow_error("coordinate_lasso(): one or more columns of y has an unbounded norm. Try rescaling.");
    } else if (b.has_nan() or b.has_inf()) throw std::runtime_error("coordinate_lasso(): initial intercepts have nan or inf. Try providing an empty array, or an array or zeros.");
    
    check_w(w, X, y);

    return coordinate_lasso_unsafe(b, w, X, y, lambda, tol, max_iter, verbose);
}

void numerics::LassoCV::fit(const arma::mat& X, const arma::vec& y) {
    check_xy(X, y, _tol);
    _dim = X.n_cols;
    KFolds2Arr<double,double> split(3);

    split.fit(X,y);
    if (_fit_intercept) _b = arma::mean(y);
    else _b = 0;
    _w = arma::zeros(X.n_cols);

    arma::rowvec bb;
    if (_fit_intercept) {
        bb.set_size(1);
        bb(0) = _b;
    }
    arma::mat ww = _w;

    auto crossval_loss = [&](double L) -> double {
        L = std::pow(10.0, L);
        double cvloss = 0;
        for (short j=0; j < 3; ++j) {
            arma::mat train_X = split.trainX(j);
            arma::mat train_y = split.trainY(j);
            arma::mat test_X = split.testX(j);
            arma::mat test_y = split.testY(j);
            if (_fit_intercept) {
                coordinate_lasso_unsafe(bb, ww, train_X, train_y, L, _tol, _max_iter, false);
                cvloss += loss(test_X, test_y, bb, ww)/3.0;
            }
            else {
                coordinate_lasso_unsafe(ww, train_X, train_y, L, _tol, _max_iter, false); // passing _w by reference for warm start
                cvloss += loss(test_X, test_y, ww)/3.0;
            }
            // if computed parameters are valid, then save, otherwise restart solver from last safe start.
            if (ww.has_nan() or ww.has_inf()) ww = _w;
            else _w = ww;

            if (_fit_intercept) {
                if (bb.has_nan() or bb.has_inf()) bb(0) = _b;
                else _b = bb(0);
            }
        }
        return cvloss;
    };
    
    _lambda = optimization::fminbnd(crossval_loss,-4,2);
    _lambda = std::pow(10.0, _lambda);
    double t = _tol / 2; // tolerance for refinement
    if (_fit_intercept) coordinate_lasso_unsafe(bb, ww, X, y, _lambda, t, _max_iter, false);
    else coordinate_lasso_unsafe(ww, X, y, _lambda, t, _max_iter, false); // final refinement

    _w = std::move(ww);
    if (_fit_intercept) _b = bb(0);
    _w.clean(t/2.0);
    _df = 0;
    for (double ww : _w) if (ww != 0) _df++;
}

arma::vec numerics::LassoCV::predict(const arma::mat& x) const {
    _check_x(x);
    return _b + x*_w;
}

double numerics::LassoCV::score(const arma::mat& x, const arma::vec& y) const {
    _check_xy(x,y);
    return r2_score(y, predict(x));
}