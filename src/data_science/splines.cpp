#include "numerics.hpp"

/* splines(m) : initialize splines object.
 * --- m : supplement fit by (m-1) degree polynomial, this parameter should be taken so m is small and (2*m - dimension of x) >= 1 */
numerics::splines::splines(int m) {
    this->m = m;
}

/* splines(lambda, m) : initialize splines object.
 * --- lambda : smoothing parameter.
 * --- m : supplement fit by (m-1) degree polynomial, this parameter should be taken so m is small and (2*m - dimension of x) >= 1 */
numerics::splines::splines(double lambda, int m) {
    this->m = m;
    this->lambda = lambda;
}

/* splines(X, Y, m) : initialize and fit splines object.
 * --- X : array of indpendent variable data, where each row is data point.
 * --- Y : array of dependent variable data, where each row is a data point.
 * --- m : supplement fit by (m-1) degree polynomial, this parameter should be taken so m is small and (2*m - dimension of x) >= 1 */
numerics::splines::splines(const arma::mat& X, const arma::mat& Y, int m) {
    dim = X.n_cols;
    n = X.n_rows;
    if (2*m - dim < 1) {
        std::cerr << "splines() warning: invalid parameter value." << std::endl
                  << "It is required that m >= " << (dim + 1)/2 << std::endl
                  << "setting m = " << std::ceil( (dim+1)/2.0) << std::endl;
        m = std::ceil( (dim+1)/2.0);
    }
    this->m = m;
    this->X = X;
    this->Y = Y;

    gen_monomials();
    arma::mat P = polyKern(X);
    arma::mat Pi = arma::pinv(P);

    arma::mat K = rbf(X);


    arma::mat V; arma::vec D;
    arma::eig_sym(D, V, K);

    
    int N = 50;
    arma::vec L = arma::logspace(-3,3,N-2);
    L = arma::join_cols(arma::vec({0,arma::datum::inf}), L);

    cv_scores = arma::zeros(N);

    for (int i=0; i<N; ++i) {
        double lam = L(i);
        if (lam == 0) {
            k_folds split(X,Y,3);
            double cv = 0;
            for (int i=0; i < 3; ++i) {
                arma::mat K0 = rbf(split.train_set_X(i), split.train_set_X(i));
                arma::mat K0i = arma::pinv(K0);
                arma::mat c_hat = K0i * split.train_set_Y(i);
                arma::mat yhat = rbf(split.train_set_X(i),split.test_set_X(i)) * c_hat;
                double dof = arma::trace(K0*K0i);
                int nn = 2*n/3;
                cv += std::pow(arma::norm(yhat - split.test_set_Y(i)),2);
            }
            cv_scores(i) += cv/n;
        } else if (std::isinf(lam)) {
            arma::mat hatmat = P*Pi;
            arma::mat yhat = hatmat*Y;
            double dof = arma::trace(hatmat);
            double cv = arma::norm(Y - yhat) / (1 - dof/n);
            cv_scores(i) = cv*cv/n;
        } else {
            arma::mat Ki = V * arma::diagmat(1/(D + lam)) * V.t();
            arma::mat c_hat = Ki * Y;
            arma::mat d_hat = Pi * (Y - K*c_hat);
            arma::mat yhat = K*c_hat + P*d_hat;
            Ki = K*Ki;
            double dof = arma::trace(P*Pi*(arma::eye(n,n) - Ki) + Ki);
            double cv = arma::norm(Y - yhat) / (1 - dof/n);
            cv_scores(i) = cv*cv/n;
        }
    }
    int I = cv_scores.index_min();
    this->lambda = L(I);
    arma::mat Ki = V * arma::diagmat(1/(D + lambda)) * V.t();
    c = Ki * Y;
    d = Pi * (Y - K*c);
    arma::mat yhat = K*c + P*d;
    Ki = K*Ki;
    df = arma::trace(P*Pi*(arma::eye(n,n) - Ki) + Ki);
    gcv = arma::norm(Y - yhat) / (1 - df/n);
    gcv *= gcv/n;
}

/* splines(X, Y, lambda, m) : initialize and fit splines object.
 * --- X : array of indpendent variable data, where each row is data point.
 * --- Y : array of dependent variable data, where each row is a data point.
 * --- lambda : smoothing parameter.
 * --- m : supplement fit by (m-1) degree polynomial, this parameter should be taken so m is small and (2*m - dimension of x) >= 1 */
numerics::splines::splines(const arma::mat& X, const arma::mat& Y, double lambda, int m) {
    if (X.n_rows != Y.n_rows) {
        std::cerr << "splines() error: number of observations in x (" << X.n_rows << ") does not equal the number of observations in y (" << Y.n_rows << ").\n";
        return;
    }
    dim = X.n_cols;
    n = X.n_rows;
    if (2*m - dim < 1) {
        std::cerr << "splines() warning: invalid parameter value." << std::endl
                  << "It is required that m >= " << (dim + 1)/2 << std::endl
                  << "setting m = " << std::ceil( (dim+1)/2.0) << std::endl;
        m = std::ceil( (dim+1)/2.0);
    }
    this->m = m;
    this->lambda = lambda;
    this->X = X;
    this->Y = Y;

    if (lambda == 0) {
        arma::mat K = rbf(X);
        arma::mat Ki = arma::pinv(K); // shouldn't happen
        c = Ki*Y;
        df = arma::trace(K*Ki);
        gcv = 0;
    } else if (std::isinf(lambda)) {
        gen_monomials();
        arma::mat P = polyKern(X);
        arma::mat Pi = arma::pinv(P);
        d = Pi*Y;
        df = arma::trace(P*Pi);
        gcv = arma::norm(Y - P*d,"fro");
        gcv *= gcv/n;
    } else {
        gen_monomials();
        arma::mat P = polyKern(X);
        arma::mat Pi = arma::pinv(P);
        arma::mat K = rbf(X);
        arma::mat Ki;
        bool chol_success = arma::inv_sympd(Ki,K + lambda*arma::eye(arma::size(K)));
        if (!chol_success) Ki = arma::pinv(K + lambda*arma::eye(arma::size(K))); // shouldn't happen
        c = Ki*Y;
        d = Pi*(Y - K*c);
        arma::mat yhat = K*c + P*d;
        Ki = K*Ki;
        df = arma::trace(P*Pi*(arma::eye(n,n) - Ki) + Ki);
        gcv = arma::norm(Y - yhat) / (1 - df/n);
        gcv *= gcv/n;
    }
}

/* splines(in) : initialize spline object by loading object from a stream. */
numerics::splines::splines(std::istream& in) {
    load(in);
}

/* fit(X, Y) : fit splines object.
 * --- X : array of indpendent variable data, where each row is data point.
 * --- Y : array of dependent variable data, where each row is a data point. */
numerics::splines& numerics::splines::fit(const arma::mat& X, const arma::mat& Y) {
    if (lambda < 0) splines(X, Y, m);
    else splines(X, Y, lambda, m);
    return *this;
}

/* fit_predict(X, Y) : fit splines object and predict on same data. same as this.fit(X,Y).predict(X) */
arma::mat numerics::splines::fit_predict(const arma::mat& X, const arma::mat& Y) {
    return fit(X,Y).predict(X);
}

/* rbf(xgrid) : build radial basis kernel matrix from fitted data evaluated at a new set of points.
 * --- xgrid : set of points to evaluate RBFs on. */
arma::mat numerics::splines::rbf(const arma::mat& xgrid) {
    int n = X.n_rows;
    int ngrid = xgrid.n_rows;
    arma::mat K = arma::zeros(ngrid, n);
    for (int i=0; i < ngrid; ++i) {
        for (int j=0; j < n; ++j) {
            double z = arma::norm(xgrid.row(i) - X.row(j));
            K(i,j) = std::pow(z, 2*m-dim);
            if (dim%2 == 0) K(i,j) *= std::log(z);
            if (std::isnan(K(i,j)) || std::isinf(K(i,j))) K(i,j) = 0;
        }
    }
    return K;
}

/* rbf(x,xgrid) : *private* build radial basis kernel from arbitrary data matrix x evaluated at xgrid. */
arma::mat numerics::splines::rbf(const arma::mat& x, const arma::mat& xgrid) {
    int n = x.n_rows;
    int ngrid = xgrid.n_rows;
    arma::mat K = arma::zeros(ngrid, n);
    for (int i=0; i < ngrid; ++i) {
        for (int j=0; j < n; ++j) {
            double z = arma::norm(xgrid.row(i) - x.row(j));
            K(i,j) = std::pow(z, 2*m-dim);
            if (dim%2 == 0) K(i,j) *= std::log(z);
            if (std::isnan(K(i,j)) || std::isinf(K(i,j))) K(i,j) = 0;
        }
    }
    return K;
}

/* polyKern(xgrid) : build polynomial basis matrix evaluated at a set of points.
 * --- xgrid : set of points to evaluate polynomial basis on. */
arma::mat numerics::splines::polyKern(const arma::mat& xgrid) {
    int n = xgrid.n_rows;
    int num_mons = monomials.size() + 1;
    arma::mat P = arma::ones(n, num_mons);
    for (int i=0; i < num_mons-1; ++i) {
        for (int r : monomials.at(i) ) {
            P.col(i+1) %= xgrid.col(r);
        }
    }
    return P;
}

/* poly_coef() : returns coefficients for polynomial basis matrix from fit. */
arma::mat numerics::splines::poly_coef() const {
    return d;
}

/* rbf_coef() : returns coefficients for RBF kernel matrix from fit. */
arma::mat numerics::splines::rbf_coef() const {
    return c;
}

/* gen_monomials() : private member function. constructs list of all monomials of requested order and dimension. */
void numerics::splines::gen_monomials() {
    std::queue<std::vector<int>> Q;
    std::set<std::vector<int>> S;
    for (int i=0; i < dim; ++i) {
        std::vector<int> str = {i};
        Q.push(str);
    }
    while ( !Q.empty() ) {
        std::vector<int> str = Q.front();
        if (str.size() > m-1) { // invalid monomial
            Q.pop();
            continue;
        }
        if (S.count(str) > 0) { // discovered
            Q.pop();
            continue;
        } else { // new node
            S.insert(str);
            for (int i=0; i < dim; ++i) {
                std::vector<int> str2 = str;
                str2.push_back(i);
                std::sort( str2.begin(), str2.end() );
                Q.push(str2);
            }
            Q.pop();
        }
    }
    monomials.clear();
    for (const std::vector<int>& str : S) monomials.push_back(str);
}

/* predict(xgrid) : evaluate spline fit on a set of new points.
 * --- xgrid : set of points to evaluate spline fit on. */
arma::mat numerics::splines::predict(const arma::mat& xgrid) {
    if (xgrid.n_cols != dim) {
        std::cerr << "splines::predict() error: dimension of new data do not match fitted data dimenstion." << std::endl
                  << "dim(fitted data) = " << dim << " =/= " << xgrid.n_cols << " = dim(new data)" << std::endl;
        return arma::mat();
    }
    if (lambda == 0) {
        return rbf(xgrid)*c;
    } else if (std::isinf(lambda)) {
        return polyKern(xgrid)*d;
    } else {
        return rbf(xgrid)*c + polyKern(xgrid)*d;
    }
}

/* splines::(xgrid) : same as predict(const arma::mat&). */
arma::mat numerics::splines::operator()(const arma::mat& xgrid) {
    return predict(xgrid);
}

/* data_X() : return independent variable data matrix. */
arma::mat numerics::splines::data_X() {
    return X;
}

/* data_Y() : return dependent variable data matrix. */
arma::mat numerics::splines::data_Y() {
    return Y;
}

/* gcv_score() : return generalized MSE estimate from leave one out cross validation. */
double numerics::splines::gcv_score() const {
    return gcv;
}

/* eff_df() : return effective degrees of freedom of spline fit. */
double numerics::splines::eff_df() const {
    return df;
}

/* smoothing_param() : returns smoothing parameter, either from initialization or from cross validation. */
double numerics::splines::smoothing_param() const {
    return lambda;
}

/* load(in) : load in object from file. */
void numerics::splines::load(std::istream& in) {
    int c_rows, c_cols, d_rows, d_cols, nm;
    in >> n >> dim >> m >> lambda >> df >> gcv;
    if (lambda = -1) lambda = arma::datum::inf;

    in >> c_rows >> c_cols;
    c = arma::zeros(c_rows,c_cols);
    for (int i=0; i < c_rows; ++i) {
        for (int j=0; j < c_cols; ++j) in >> c(i,j);
    }

    in >> d_rows >> d_cols;
    d = arma::zeros(d_rows, d_cols);
    for (int i=0; i < d_rows; ++i) {
        for (int j=0; j < d_cols; ++j) in >> d(i,j);
    }

    X = arma::zeros(n,dim);
    for (int i=0; i < n; ++i) {
        for (int j=0; j < dim; ++j) in >> X(i,j);
    }

    monomials.clear();
    while (true) {
        in >> nm;
        if (in.eof()) break;
        std::vector<int> str(nm);
        for (int i=0; i < nm; ++i) {
            in >> str.at(i);
        }
        monomials.push_back(str);
    }
}

/* save(out) : save object to file. */
void numerics::splines::save(std::ostream& out) {
    out << std::setprecision(10);
    // parameters in order : n, dim, m, lambda, df, gcv
    out << n << " " << dim << " "
        << m << " ";
    if (std::isinf(lambda)) out << -1 << " ";
    else out << lambda << " ";
    out << df << " " << gcv << std::endl << std::endl;

    // c_rows, c_cols, c
    out << c.n_rows << " " << c.n_cols;
    for (int i=0; i < c.n_rows; ++i) {
        for (int j=0; j < c.n_cols; ++j) out << " " << c(i,j);
        out << std::endl;
    }

    // d_rows, d_cols, d
    out << d.n_rows << " " << d.n_cols;
    for (int i=0; i < d.n_rows; ++i) {
        for (int j=0; j < d.n_cols; ++j) out << " " << d(i,j);
        out << std::endl;
    }

    // X
    for (int i=0; i < n; ++i) {
        for (int j=0; j < dim; ++j) out << " " << X(i,j);
        out << std::endl;
    }

    // monomials
    for (std::vector<int>& str : monomials) {
        out << str.size() << " ";
        for (int i : str) out << " " << i;
        out << std::endl;
    }
}