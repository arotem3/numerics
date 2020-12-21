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

void fit_logreg(arma::mat& w, const arma::sp_mat& I, const arma::mat& P, const arma::mat& y, double L) {
    u_long n = P.n_cols-1;

    auto f = [&](const arma::vec& coefs) -> double {
        arma::mat W = arma::reshape(coefs, n+1, y.n_cols);

        arma::mat yh = P * W;
        numerics::softmax_inplace(yh);

        return 0.5*L*arma::trace(W.t()*I*W) - arma::accu(yh % y);
    };

    auto grad_f = [&](const arma::vec& coefs) -> arma::vec {
        arma::mat W = arma::reshape(coefs, n+1, y.n_cols);
        
        arma::mat yh = P * W;
        numerics::softmax_inplace(yh);

        arma::mat df = L*I*W - P.t()*(y - yh);

        return df.as_col();
    };

    arma::vec coefs = w.as_col();

    numerics::optimization::TrustMin fmin;
    fmin.minimize(coefs, f, grad_f);

    w = arma::reshape(coefs, n+1, y.n_cols);
}

arma::mat pred_logreg(const arma::mat& w, const arma::mat& P) {
    arma::mat yh = P*w;
    numerics::softmax_inplace(yh);
    return yh;
}

void numerics::LogisticRegression::fit(const arma::mat& x, const arma::uvec& y) {
    _check_xy(x, y);
    _dim = x.n_cols;
    u_long nobs = x.n_rows;

    _encoder.fit(y);
    arma::mat onehot = _encoder.encode(y);

    u_long n = x.n_cols;
    arma::mat P(nobs, n+1);
    arma::sp_mat I = arma::speye(n+1,n+1); I(0,0) = 0;
    P.col(0).ones();
    P.cols(1,n) = x;

    arma::mat W = arma::zeros(n+1, onehot.n_cols);

    if (_lambda < 0) {
        u_short nfolds = 5;
        if (nobs / nfolds < 50) nfolds = 3;
        KFolds2Arr<double,double> split(nfolds);
        split.fit(P, onehot);

        auto cv = [&](double L) -> double {
            L = std::pow(10.0, L);
            double score = 0;
            for (u_short i=0; i < nfolds; ++i) {
                fit_logreg(W, I, split.trainX(i), split.trainY(i), L);
                arma::mat p = pred_logreg(W, split.testX(i));
                score -= accuracy_score(_encoder.decode(split.testY(i)), _encoder.decode(p));
            }
            return score;
        };
        _lambda = optimization::fminbnd(cv, -5, 4, 0.1);
        _lambda = std::pow(10.0, _lambda);
    }

    fit_logreg(W, I, P, onehot, _lambda);
    _b = W.row(0);
    _w = W.rows(1,n);
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