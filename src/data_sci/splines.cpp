#include "numerics.hpp"

numerics::splines::splines() {
    // DO NOTHING
}

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
    arma::mat Q,R;
    arma::qr(Q,R,P);
    arma::mat Pinv = arma::solve(R, Q.t());
    Q = Q.cols(dim+1, n-1);

    arma::mat K = rbf(X);
    
    int N = 50;
    arma::vec L = arma::logspace(-3,3,N-2);
    L = arma::join_cols(arma::vec({0,arma::datum::inf}), L);

    arma::vec GCV = arma::zeros(N);

    for (int i=0; i<N; ++i) {
        this->lambda = L(i);
        fit(K, P, Q, Pinv);
        GCV(i) = gcv;
    }
    int I = GCV.index_min();
    this->lambda = L(I);
    fit(K, P, Q, Pinv);
}

numerics::splines::splines(const arma::mat& X, const arma::mat& Y, double lambda, int m) {
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

    gen_monomials();
    arma::mat P = polyKern(X);
    arma::mat Q,R;
    arma::qr(Q,R,P);
    arma::mat Pinv = arma::solve(R, Q.t());
    Q = Q.cols(dim+1, n-1);

    arma::mat HatMat = P * Pinv;
    arma::mat K = rbf(X);
    fit(K,P,Q,Pinv);
}

numerics::splines::splines(std::istream& in) {
    load(in);
}

void numerics::splines::fit(arma::mat& K, arma::mat& P, arma::mat& Q, arma::mat& Pinv) {
    arma::mat HatMat;
    if (std::isinf(lambda)) {
        c = arma::zeros(n);
        d = Pinv * Y;
        HatMat = P * Pinv;
    } else {
        arma::mat KHC = arma::solve(Q.t() * K * Q + lambda * arma::eye(n-dim-1,n-dim-1), Q.t() );
        KHC = Q*KHC;
        c = KHC*Y;

        d = Pinv * (Y - K*c);

        KHC = K * KHC;
        HatMat = (P * Pinv) * (arma::eye(arma::size(KHC)) - KHC) + KHC;
    }

    df = arma::trace(HatMat);

    arma::vec yHat = HatMat * Y;
    gcv = arma::norm(Y - yHat) * n / (n - df);
    gcv *= gcv;
}

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

arma::vec numerics::splines::poly_coef() const {
    return d;
}

arma::vec numerics::splines::rbf_coef() const {
    return c;
}

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

arma::mat numerics::splines::predict(const arma::mat& xgrid) {
    arma::mat K = rbf(xgrid);
    arma::mat P = polyKern(xgrid);
    return K*c + P*d;
}

arma::mat numerics::splines::operator()(const arma::mat& xgrid) {
    return predict(xgrid);
}

arma::mat numerics::splines::data_X() const {
    return X;
}

arma::mat numerics::splines::data_Y() const {
    return Y;
}

double numerics::splines::gcv_score() const {
    return gcv;
}

double numerics::splines::eff_df() const {
    return df;
}

double numerics::splines::smoothing_param() const {
    return lambda;
}

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