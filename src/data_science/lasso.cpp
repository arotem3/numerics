#include <numerics.hpp>

/* soft_threshold(u,v) : __utility for coordinate_lasso__ compute the soft threshold of u parameterized by v. i.e. sign(u) * positive_part(|u| - v) */
inline arma::mat soft_threshold(const arma::mat& u, double v) {
    return arma::sign(u)%(arma::abs(u)-v > 0)%(arma::abs(u)-v);
}

/* iteration_verbatim(iter, fval) : __utility for coordinate_lasso__ print the current iteration of coordinate descent to terminal. */
inline void iteration_verbatim(int iter, double fval) {
    std::cout << "|" << std::right << std::setw(6) << std::setfill(' ') << iter
              << "|" << std::scientific << std::setprecision(4) << std::right << std::setw(12) << std::setfill(' ') << fval
              << "|\n";
}

/* compute_rho(y, X, w, i, p) : __utility for coordinate_lasso__ partial computation of the coordinate descent update. */
arma::rowvec compute_rho(const arma::mat& y, const arma::mat& X, const arma::mat& w, int i, int p) {
    arma::mat yhat = arma::zeros(arma::size(y));
    for (int j=0; j < p; ++j) {
        if (j==i) continue;
        else {
            yhat += X.col(j) * w.row(j);
        }
    }
    return X.col(i).t()*(y - yhat);
}

/* coordinate_lasso(y,X,w,lambda,first_term_intercept=false, tol=1e-5, max_iter=1000, verbose=false) : solves the lasso linear regression problem ||y-X*w||_2^2 + lambda*||w||_1 using coordinate descent.
 * --- y : dependent variable.
 * --- X : independent variable.
 * --- w : weights.
 * --- lambda : L1 regularization parameter.
 * --- first_term_intercerpt : if true, the method treats the first coloumn of X as an intercept term and does not apply the lasso regularization to it.
 * --- tol : tolerance for convergence, coordinate descent stops when |x_old - x_new| < tol.
 * --- max_iter : maximum number of iterations before premature stopping.
 * --- verbose : [false] no printing, [true] print each iteration.
 * returns [0] successful convergence [1] maximum iterations reach. */
int numerics::coordinate_lasso(const arma::mat& y, const arma::mat& X, arma::mat& w, double lambda, bool first_term_intercept, double tol, uint max_iter, bool verbose) {
    int p = X.n_cols;
    arma::vec z = arma::zeros(p);
    for (int i=0; i < p; ++i) {
        z(i) = std::pow(arma::norm(X.col(i)),2);
    }

    if (verbose) {
        std::cout << "|" << std::right << std::setw(6) << std::setfill(' ') << "iter"
                  << "|" << std::right << std::setw(12) << std::setfill(' ') << "RMSE"
                  << "|" << std::right << std::setw(12) << std::setfill(' ') << "dw"
                  << "|\n";
    }
    
    uint n=0;
    arma::mat w_prev;
    do {
        w_prev = w;
        for (int i=0; i < p; ++i) {
            n++;
            arma::rowvec rho = compute_rho(y,X,w,i,p);
            if (first_term_intercept && i==0) w.row(i) = rho/z(i);
            else w.row(i) = soft_threshold(rho, lambda/2)/z(i);

            if (verbose) iteration_verbatim(n, arma::norm(y - X*w,"fro")/(X.n_rows));
            if (n >= max_iter) {
                if (verbose) std::cout << "---maximum number of iterations reached---\n";
                return 1;
            }
        }
    } while (arma::norm(w-w_prev,"inf") > tol*arma::norm(w,"inf"));
    if (verbose) std::cout << "---converged to solution within tolerance---\n";
    return 0;
}

/* lasso_cv(tol=1e-5, max_iter=1000) : initialize cross validating lasso regression object.
 * --- tol : tolerance for convergence, coordinate descent stops when |x_old - x_new| < tol.
 * --- max_iter : maximum number of iterations before premature stopping. */
numerics::lasso_cv::lasso_cv(double tol, uint max_iter) : regularizing_param(_lambda), RMSE(_rmse), coef(_w), residuals(_res), eff_df(_df) {
    _tol = tol;
    _max_iter = max_iter;
}

/* fit(X, y, first_term_intercept=false) : fit by cross-validation which benefits from a "warm-start" for the paramter estimates using the solution from the previous evaluation.
 * --- X : independent variable.
 * --- y : dependent variable.
 * --- first_term_intercerpt : if true, the method treats the first coloumn of X as an intercept term and does not apply the lasso regularization to it. */
void numerics::lasso_cv::fit(const arma::mat& X, const arma::mat& y, bool first_term_intercept) {
    _w = arma::zeros(X.n_cols, y.n_cols);
    numerics::k_folds train_test(X,y,2);
    arma::mat train_X = train_test.train_set_X(0), train_y = train_test.train_set_Y(0), test_x = train_test.test_set_X(0), test_y = train_test.test_set_Y(0);

    auto crossval_loss = [&](double L) -> double {
        coordinate_lasso(train_y,train_X,_w,L,first_term_intercept,_tol,_max_iter, false);
        return arma::norm(test_y - test_x*_w, "fro");
    };
    _lambda = numerics::fminbnd(crossval_loss,1e-3,10);
    _res = y - X*_w;
    _rmse = arma::norm(_res, "fro") / _res.n_elem;
    arma::uvec where_zero = arma::all(arma::abs(_w) < _tol, 1);
    _df = X.n_cols - arma::sum(where_zero);
    where_zero = arma::find(where_zero);
    _w.rows(where_zero).fill(0);
}