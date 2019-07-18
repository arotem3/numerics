#include <numerics.hpp>

/* regularizer(lambda) : initialize regularizer object.
 * --- lambda : regularizing parameter for L2-regularization. default lambda=nan, this default tells the object to select a regularization parameter by 10 fold cross validation. */
numerics::regularizer::regularizer(double lambda) {
    if (lambda < 0 || std::isinf(lambda)) {
        std::cerr << "regularizer error: invalid choice of lambda (your lambda = "<< lambda << ")." << std::endl
                  << "Appropriate range: 0 <= lambda < inf or lambda=nan" << std::endl
                  << "setting lambda = nan, which will allow the regularizer to select a lambda by cross validation." << std::endl;
        this->lambda = arma::datum::nan;
    } else this->lambda = lambda;
    use_L2 = true;
}

/* regularizer(regular_mat, lambda) : initialize regularizer object with custom regularization matrix.
 * --- regular_mat : regularizing matrix. Caveat, must be square and symmetric n x n with n = X.n_cols. There will be no internal check for symmetry, though.
 * --- lambda : regularizing parameter for L2-regularization. default lambda=nan, this default tells the object to select a regularization parameter by 10 fold cross validation. */
numerics::regularizer::regularizer(const arma::mat& regular_mat, double lambda) {
    if (!regular_mat.is_square()) {
        std::cerr << "regularizer error: choice of regularizing matrix must be square (your size(regular_mat) = " << arma::size(regular_mat) << ")." << std::endl
                  << "trying instead A.t() * A, where A = regular_mat." << std::endl
                  << "note: this may still cause an error while fitting if regular_mat.n_cols != X.n_cols" << std::endl;
        this->regular_mat = regular_mat.t() * regular_mat;
    } else this->regular_mat = regular_mat;
    if (lambda < 0 || std::isinf(lambda)) {
        std::cerr << "regularizer error: invalid choice of lambda (your lambda = "<< lambda << ")." << std::endl
                  << "Appropriate range: 0 <= lambda < inf or lambda=nan" << std::endl
                  << "setting lambda = nan, which will allow the regularizer to select a lambda by cross validation." << std::endl;
        this->lambda = arma::datum::nan;
    } else this->lambda = lambda;
    use_L2 = false;
}

/* regularizer(x, y, lambda, use_conj_grad) : initialize and fit regularizer object.
 * --- x : linear basis to fit such that y = X*c.
 * --- y : output data to predict.
 * --- lambda : regularizing parameter for L2-regularization. default lambda=nan, this default tells the object to select a regularization parameter by 10 fold cross validation.
 * --- use_conj_grad : use conjugate gradient to solve. */
numerics::regularizer::regularizer(const arma::mat& x, const arma::mat& y, double lambda) {
    if (x.n_rows != y.n_rows) { // error
        std::cerr << "regularizer error: number of independent variable observations must equal the number of dependent variable observations." << std::endl
                  << "X.n_rows = " << x.n_rows << " != " << y.n_rows << " = Y.n_rows" << std::endl;
        return;
    }
    this->lambda = lambda;
    use_L2 = true;
    regular_mat = arma::eye(x.n_cols, x.n_cols);

    if (std::isnan(lambda)) cross_validate(x,y);
    fit_no_replace(x, y, this->lambda);
}

/* regularizer(x, y, regular_mat, lambda, use_conj_grad) : initialize and fit regularizer object.
 * --- X : linear basis to fit such that y = X*c.
 * --- Y : output data to predict.
 * --- regular_mat : regularizing matrix. Caveat, must be square and symmetric n x n with n = X.n_cols. There will be no internal check for symmetry, though.
 * --- lambda : regularizing parameter for L2-regularization. default lambda=nan, this default tells the object to select a regularization parameter by 10 fold cross validation.
 * --- use_conj_grad : use conjugate gradient to solve. */
numerics::regularizer::regularizer(const arma::mat& x, const arma::mat& y, const arma::mat& regular_mat, double lambda, bool use_conj_grad) {
    if (regular_mat.n_rows != x.n_rows || regular_mat.n_cols != x.n_rows) {
        std::cerr << "regularizer error: choice of regularizing matrix is innapropriate. size(regular_mat) should be n x n where n = X.n_cols" << std::endl
                  << "your size(regular_mat) = " << arma::size(regular_mat) << " != " << x.n_cols << " " << x.n_cols << " = n x n" << std::endl;
        return;
    }
    this->regular_mat = regular_mat;
    this->lambda = lambda;
    use_L2 = false;
    use_cgd = use_conj_grad;

    if (std::isnan(lambda)) cross_validate(x,y);
    fit_no_replace(x, y, this->lambda);
}

/* cross_validate(X, Y) : *private* performs cross validation with respect to lambda. */
void numerics::regularizer::cross_validate(const arma::mat& X, const arma::mat& Y) {
    uint num_folds = 5;
    if (X.n_rows / num_folds < 10) {
        num_folds = 3;
        if (X.n_rows / num_folds < 10) {
            num_folds = 2;
        }
    }
    k_folds split(X, Y, num_folds);
    
    int N = 30;
    arma::vec L = arma::logspace(-5,3,N);

    arma::vec cv_scores = arma::zeros(N);
    for (uint j=0; j < num_folds; ++j) {
        arma::mat V, R, xx = split.not_fold_X(j), yy = split.not_fold_Y(j);
        arma::vec D;
        if (use_L2) {
            arma::eig_sym(D, V, xx.t()*xx);
            arma::mat R = V.t() * regular_mat * V;
        }
        double score = 0;
        #pragma omp parallel for
        for (int i=0; i < N; ++i) {
            arma::mat c;
            if (use_L2) c = V * arma::diagmat(1/(D + L(i))) * V.t() * xx.t() * yy;
            else {
                arma::mat RHS = xx.t()*xx + L(i)*regular_mat, LHS = xx.t() * yy;
                if (use_cgd) cgd(RHS, LHS, c);
                else c = arma::solve(RHS, LHS);
            }
            score += arma::norm(split.fold_Y(j) - split.fold_X(j)*c);
        }
        cv_scores(j) = score/X.n_rows;
    }
    int indmin = arma::index_min(cv_scores);
    lambda = L(indmin);
}

/* fit_no_replace(x, y, lam) : *private* perform fitting with out replacing private members */
void numerics::regularizer::fit_no_replace(const arma::mat& x, const arma::mat& y, double lam) {
    arma::mat A = x.t() * x + lam*regular_mat, b = x.t();
    arma::mat hatmat = arma::zeros(A.n_cols, b.n_cols);
    if (use_cgd) cgd(A,b,hatmat);
    else hatmat = arma::solve(A, b);
    coefs = hatmat * y;
    hatmat = x * hatmat;
    df = arma::trace(hatmat);
    arma::mat y_hat = hatmat * y;
    cv = arma::norm(y - y_hat,"fro");
    cv *= cv/x.n_rows;
}

/* fit(x, y, use_conj_grad) : fit regularizer object. Method of fit depends on how the object is initialized.
 * --- x : linear basis to fit such that y = X*c.
 * --- y : output data to predict.
 * --- use_conj_grad : use conjugate gradient to solve. */
arma::mat numerics::regularizer::fit(const arma::mat& x, const arma::mat& y, bool use_conj_grad) {
    bool use_cgd = use_conj_grad;
    if (x.n_rows != y.n_rows) { // error
        std::cerr << "regularizer::fit() error: number of independent variable observations must equal the number of dependent variable observations." << std::endl
                    << "X.n_rows = " << x.n_rows << " != " << y.n_rows << " = Y.n_rows" << std::endl
                    << "object will not be fitted." << std::endl;
    } else {
        if (use_L2) regular_mat = arma::eye(x.n_cols, x.n_cols);
        if (std::isnan(lambda)) cross_validate(x, y);
        fit_no_replace(x, y, lambda);
    }
    return coefs;
}

/* coef() : returns coefficients for linear basis found from fit. */
arma::mat numerics::regularizer::coef() {
    return coefs;
}

/* regularizing_mat() : return regularizing matrix */
arma::mat numerics::regularizer::regularizing_mat() const {
    return regular_mat;
}

/* MSE() : returns MSE from fit. */
double numerics::regularizer::MSE() const {
    return cv;
}

/* eff_df() : returns effictive degrees of freedom from fit. */
double numerics::regularizer::eff_df() const {
    return df;
}

/* regularizing_param() : return parameter used for L2-regularization. If custom regularizing matrix is used nan will be returned. */
double numerics::regularizer::regularizing_param() const {
    return lambda;
}