#include "numerics.hpp"

/* REGULARIZER : initialize regularizer object.
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

/* REGULARIZER : initialize regularizer object with custom regularization matrix.
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

/* REGULARIZER : initialize and fit regularizer object.
 * --- X : linear basis to fit such that y = X*c.
 * --- Y : output data to predict.
 * --- lambda : regularizing parameter for L2-regularization. default lambda=nan, this default tells the object to select a regularization parameter by 10 fold cross validation.
 * --- use_conj_grad : use conjugate gradient to solve. */
numerics::regularizer::regularizer(const arma::mat& x, const arma::mat& y, double lambda, bool use_conj_grad) {
    if (x.n_rows != y.n_rows) { // error
        std::cerr << "regularizer error: number of independent variable observations must equal the number of dependent variable observations." << std::endl
                  << "X.n_rows = " << x.n_rows << " != " << y.n_rows << " = Y.n_rows" << std::endl;
        return;
    }
    X = x;
    Y = y;
    n_obs = X.n_rows;
    x_dim = X.n_cols;
    y_dim = Y.n_cols;
    this->lambda = lambda;
    use_L2 = true;
    regular_mat = arma::eye(x_dim, x_dim);
    use_cgd = use_conj_grad;

    if (std::isnan(lambda)) cross_validate();
    fit_no_replace(X, Y, this->lambda);
}

/* REGULARIZER : initialize and fit regularizer object.
 * --- X : linear basis to fit such that y = X*c.
 * --- Y : output data to predict.
 * --- regular_mat : regularizing matrix. Caveat, must be square and symmetric n x n with n = X.n_cols. There will be no internal check for symmetry, though.
 * --- lambda : regularizing parameter for L2-regularization. default lambda=nan, this default tells the object to select a regularization parameter by 10 fold cross validation.
 * --- use_conj_grad : use conjugate gradient to solve. */
numerics::regularizer::regularizer(const arma::mat& x, const arma::mat& y, const arma::mat& regular_mat, double lambda, bool use_conj_grad) {
    if (regular_mat.n_rows != x_dim || regular_mat.n_cols != x_dim) {
        std::cerr << "regularizer error: choice of regularizing matrix is innapropriate. size(regular_mat) should be n x n where n = X.n_cols" << std::endl
                  << "your size(regular_mat) = " << arma::size(regular_mat) << " != " << X.n_cols << " " << X.n_cols << " = n x n" << std::endl;
        return;
    }
    X = x;
    Y = y;
    n_obs = X.n_rows;
    x_dim = X.n_cols;
    y_dim = Y.n_cols;
    this->regular_mat = regular_mat;
    this->lambda = lambda;
    use_L2 = false;
    use_cgd = use_conj_grad;

    if (std::isnan(lambda)) cross_validate();
    fit_no_replace(X,Y, this->lambda);
}

/* REGULARIZER : initilize regularizer object from saved instance in stream.
 * --- in : stream where regularizer instance is stored, such as a file stream. */
numerics::regularizer::regularizer(std::istream& in) {
    load(in);
}

/* CROSS_VALIDATE : *private* performs cross validation with respect to lambda. */
void numerics::regularizer::cross_validate() {
    uint num_folds = 10;
    if (n_obs / num_folds < 10) {
        num_folds = 5;
        if (n_obs / num_folds < 10) {
            num_folds = 3;
        }
    }
    folds F = k_fold(X, Y, num_folds);
    
    int N = 50;
    arma::vec L = arma::logspace(-5,3,N);

    arma::vec cv_scores = arma::zeros(N);
    for (int i=0; i < N; ++i) {
        for (uint j=0; j < num_folds; ++j) {
            fit_no_replace(
                X.rows(F.at(j).exclude_indices),
                Y.rows(F.at(j).exclude_indices),
                L(i)
            );
            cv_scores(i) = cv;
        }
    }
    int indmin = arma::index_min(cv_scores);
    lambda = L(indmin);
}

/* FIT_NO_REPLACE : *private* perform fitting with out replacing private members */
void numerics::regularizer::fit_no_replace(const arma::mat& x, const arma::mat& y, double lam) {
    arma::mat A = x.t() * x + lam*regular_mat, b = x.t();
    arma::mat hatmat = arma::zeros(A.n_cols, b.n_cols);
    if (use_cgd) cgd(A,b,hatmat);
    else hatmat = arma::solve(A, b);
    coefs = hatmat * y;
    hatmat = x * hatmat;
    df = arma::trace(hatmat);
    y_hat = hatmat * y;
    cv = arma::norm(y - y_hat,"fro");
    cv *= cv/n_obs;
}

/* FIT : fit regularizer object. Method of fit depends on how the object is initialized.
 * --- x : linear basis to fit such that y = X*c.
 * --- y : output data to predict.
 * --- use_conj_grad : use conjugate gradient to solve. */
numerics::regularizer& numerics::regularizer::fit(const arma::mat& x, const arma::mat& y, bool use_conj_grad) {
    if (x.n_rows != y.n_rows) { // error
        std::cerr << "regularizer::fit() error: number of independent variable observations must equal the number of dependent variable observations." << std::endl
                    << "X.n_rows = " << x.n_rows << " != " << y.n_rows << " = Y.n_rows" << std::endl
                    << "object will not be fitted." << std::endl;
    } else {
        X = x;
        Y = y;
        use_cgd = use_conj_grad;
        n_obs = X.n_rows;
        x_dim = X.n_cols;
        y_dim = Y.n_cols;
        if (use_L2) regular_mat = arma::eye(x_dim, x_dim);
        if (std::isnan(lambda)) cross_validate();
        fit_no_replace(X, Y, lambda);
    }
    return *this;
}

/* FIT_PREDICT : fit regularizer object and predict on same data. Method of fit depends on how the object is initialized.
 * --- X : linear basis to fit such that y = X*c.
 * --- Y : output data to predict.
 * --- use_conj_grad : use conjugate gradient to solve. */
arma::mat numerics::regularizer::fit_predict(const arma::mat& X, const arma::mat& Y, bool use_conj_grad) {
    fit(X,Y, use_conj_grad);
    return y_hat;
}

/* PREDICT : predict on new input data. It is necessary that xgrid.n_cols == X.n_cols
 * --- X : linear basis to fit such that y = X*c. */
arma::mat numerics::regularizer::predict(const arma::mat& xgrid) {
    if (xgrid.n_cols != x_dim) {
        std::cerr << "regularizer::predict() error: xgrid.n_cols != X.n_cols from fit." << std::endl
                  << "your xgrid.n_cols = " << xgrid.n_cols << " != " << X.n_cols << " = X.n_cols" << std::endl;
        return arma::mat();
    }
    return xgrid * coefs;
}

/* PREDICT : predict on same data as fit. */
arma::mat numerics::regularizer::predict() {
    return y_hat;
}

/* OPERATOR() : same as predict(xgrid). */
arma::mat numerics::regularizer::operator()(const arma::mat& xgrid) {
    return predict(xgrid);
}

arma::mat numerics::regularizer::operator()() {
    return predict();
}

/* DATA_X : returns linear basis data used to fit object. */
arma::mat numerics::regularizer::data_X() {
    return X;
}

/* DATA_Y : returns output data used to fit object. */
arma::mat numerics::regularizer::data_Y() {
    return Y;
}

/* REGULARIZING_MATRIX : returns regularizing matrix used to fit object. */
arma::mat numerics::regularizer::regularizing_matrix() {
    return regular_mat;
}

/* COEF : returns coefficients for linear basis found from fit. */
arma::mat numerics::regularizer::coef() {
    return coefs;
}

/* MSE : returns MSE from fit. */
double numerics::regularizer::MSE() const {
    return cv;
}

/* EFF_DF : returns effictive degrees of freedom from fit. */
double numerics::regularizer::eff_df() const {
    return df;
}

/* REGULARIZING_PARAM : return parameter used for L2-regularization. If custom regularizing matrix is used nan will be returned. */
double numerics::regularizer::regularizing_param() const {
    return lambda;
}

/* SAVE : save object to stream.
 * --- out : stream to save object to. */
void numerics::regularizer::save(std::ostream& out) {
    out << n_obs << " " << x_dim << " " << y_dim << " " << lambda << " " << cv << " " << df << std::endl;
    X.raw_print(out);
    Y.raw_print(out);
    coefs.raw_print(out);
    regular_mat.raw_print(out);
}

/* LOAD : load object from stream.
 * --- in : stream to load object from which an instance of a regularizer object is stored. */
void numerics::regularizer::load(std::istream& in) {
    in >> n_obs >> x_dim >> y_dim >> lambda >> cv >> df;
    X = arma::zeros(n_obs, x_dim);
    Y = arma::zeros(n_obs, y_dim);
    coefs = arma::zeros(x_dim, y_dim);
    regular_mat = arma::zeros(x_dim, x_dim);
    for (uint i=0; i < n_obs; ++i) {
        for (uint j=0; j < x_dim; ++j) {
            in >> X(i,j);
        }
    }

    for (uint i=0; i < n_obs; ++i) {
        for (uint j=0; j < y_dim; ++j) {
            in >> Y(i,j);
        }
    }

    for (uint i=0; i < x_dim; ++i) {
        for (uint j=0; j < y_dim; ++j) {
            in >> coefs(i,j);
        }
    }

    for (uint i=0; i < x_dim; ++i) {
        for (uint j=0; j < x_dim; ++j) {
            in >> regular_mat(i,j);
        }
    }

    use_cgd = false;
    use_L2 = false;
    y_hat = X*coefs;
}