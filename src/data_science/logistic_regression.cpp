#include <numerics.hpp>

void mult_coefs_serial(arma::mat& yh, const arma::vec& coefs, const arma::rowvec& b, const arma::mat& w, const arma::mat& x) {
    yh = arma::repmat(coefs.subvec(0,b.n_elem-1).as_row(), x.n_rows, 1);
    arma::uword start = b.n_elem;
    for (u_long i=0; i < w.n_rows; ++i) {
        for (u_long j=0; j < w.n_cols; ++j) {
            arma::uword ij = start + arma::sub2ind(arma::size(w), i, j);
            yh.col(j) += x.col(i) * coefs(ij);
        }
    }
    numerics::softmax_inplace(yh);
}

arma::vec grad_serial(const arma::mat& yh, const arma::vec& coefs, const arma::rowvec& b, const arma::mat& w, const arma::mat& x, const arma::mat& y, double L) {
    u_long size = x.n_cols + 1;
    size *= y.n_cols;

    arma::vec out(size);

    arma::mat res = y - yh;

    for (u_long i=0; i < b.n_elem; ++i) {
        out(i) = -arma::sum(res.col(i));
    }

    arma::uword start = b.n_elem;
    for (u_long i=0; i < w.n_rows; ++i) {
        for (u_long j=0; j < w.n_cols; ++j) {
            arma::uword ij = start + arma::sub2ind(arma::size(w), i, j);
            out(ij) = -arma::dot(x.col(i), res.col(j));
            out(ij) += L * coefs(ij);
        }
    }
    return out;
}

void fit_logreg(arma::rowvec& b, arma::mat& w, const arma::mat& x, const arma::mat& y, double L) {
    u_long size = b.n_elem + w.n_elem;

    arma::vec p = arma::zeros(size);
    p(arma::span(0, b.n_elem-1)) = b.as_col();
    p(arma::span(b.n_elem, b.n_elem + w.n_elem - 1)) = w.as_col();

    auto f = [&](const arma::vec& coefs) -> double {
        arma::mat yh;
        mult_coefs_serial(yh, coefs, b, w, x);
        double cnorm = 0;
        arma::uword start = b.n_elem;
        for (u_long i=start; i < coefs.n_elem; ++i) cnorm += std::pow(coefs(i),2);

        return 0.5*L*cnorm - arma::accu(yh % y);
    };

    auto grad_f = [&](const arma::vec& coefs) -> arma::vec {
        arma::mat yh;
        mult_coefs_serial(yh, coefs, b, w, x);
        return grad_serial(yh, coefs, b, w, x, y, L);
    };

    numerics::optimization::LBFGS fmin;
    fmin.minimize(p, f, grad_f);

    b = p.subvec(0, b.n_elem-1).as_row();
    w = arma::reshape(p.subvec(b.n_elem, b.n_elem + w.n_elem - 1), arma::size(w));
}

arma::mat pred_logreg(const arma::rowvec& b, const arma::mat& w, const arma::mat& x) {
    arma::mat yh = x*w;
    yh.each_row() += b;
    numerics::softmax_inplace(yh);
    return yh;
}

void numerics::LogisticRegression::fit(const arma::mat& x, const arma::uvec& y) {
    _check_xy(x, y);
    _dim = x.n_cols;
    u_long nobs = x.n_rows;

    _encoder.fit(y);
    arma::mat onehot = _encoder.encode(y);

    _b = arma::zeros(1, onehot.n_cols);
    _w = arma::zeros(x.n_cols, onehot.n_cols);

    if (_lambda < 0) {
        u_short nfolds = 5;
        if (nobs / nfolds < 50) nfolds = 3;
        KFolds2Arr<double,double> split(nfolds);
        split.fit(x, onehot);

        auto cv = [&](double L) -> double {
            L = std::pow(10.0, L);
            double score = 0;
            for (u_short i=0; i < nfolds; ++i) {
                fit_logreg(_b, _w, split.trainX(i), split.trainY(i), L);
                arma::mat p = pred_logreg(_b, _w, split.testX(i));
                score -= accuracy_score(_encoder.decode(split.testY(i)), _encoder.decode(p));
            }
            return score;
        };
        _lambda = optimization::fminbnd(cv, -5, 4, 0.1);
        _lambda = std::pow(10.0, _lambda);
    }

    fit_logreg(_b, _w, x, onehot, _lambda);
}

arma::mat numerics::LogisticRegression::predict_proba(const arma::mat& x) const {
    _check_x(x);
    arma::mat yh = x * _w;
    yh.each_row() += _b;
    softmax_inplace(yh);
    return yh;
}

arma::uvec numerics::LogisticRegression::predict(const arma::mat& x) const {
    return _encoder.decode(predict_proba(x));
}

double numerics::LogisticRegression::score(const arma::mat& x, const arma::uvec& y) const {
    _check_xy(x,y);
    return accuracy_score(y, predict(x));
}