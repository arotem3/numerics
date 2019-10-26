#include <numerics.hpp>

numerics::logistic_regression::logistic_regression(std::istream& in) {
    load(in);
}

void numerics::logistic_regression::load(std::istream& in) {
    int n_vars, n_cats;
    in >> n_obs >> n_cats >> n_vars >> lambda >> beta;
    x = arma::zeros(n_obs,n_vars);
    for (int i=0; i < n_vars; ++i) {
        for (int j=0; j < n_obs; ++j) {
            in >> x(j,i);
        }
    }
    y = arma::zeros(n_obs,n_cats);
    for (int i=0; i < n_cats; ++i) {
        for (int j=0; j < n_obs; ++j) {
            in >> y(j,i);
        }
    }
    c = arma::zeros(n_vars+1,n_cats);
    for (int i=0; i < n_cats; ++i) {
        for (int j=0; j < n_vars; ++j) {
            in >> c(j,i);
        }
    }
    d = arma::zeros(n_obs, n_cats);
    for (int i=0; i < n_cats; ++i) {
        for (int j=0; j < n_obs; ++j) {
            in >> d(j,i);
        }
    }
    L = arma::zeros(25);
    cv_scores = arma::zeros(25);
    for (int i=0; i < 25; ++i) in >> L(i);
    for (int i=0; i < 25; ++i) in >> cv_scores(i);
}

void numerics::logistic_regression::save(std::ostream& out) {
    int n_vars = x.n_cols, n_cats = y.n_cols;
    out << n_obs << " " << n_cats << " " << n_vars << " " << lambda << " " << beta << std::endl;
    x.t().raw_print(out);
    y.t().raw_print(out);
    c.t().raw_print(out);
    d.t().raw_print(out);
    L.t().raw_print(out);
    cv_scores.t().raw_print(out);
}

arma::mat numerics::logistic_regression::softmax(const arma::mat& z) {
    arma::mat p = arma::exp(z - arma::max(z,1));
    p.each_row([](arma::rowvec& r)->void{r /= arma::accu(r);});
    return p;
}

arma::mat numerics::logistic_regression::rbf(const arma::mat& xgrid) {
    int n = x.n_rows;
    arma::mat B = arma::zeros(xgrid.n_rows,n);
    arma::mat R(xgrid.n_rows, xgrid.n_cols);
    for (int i=0; i < n; ++i) {
        R = xgrid;
        R.each_row([&](arma::rowvec& q)->void{q -= x.row(i); q %= q;}); // R = (xgrid - x(i))^2
        B.col(i) = arma::exp(-arma::sum(R,1)/beta);
    }
    return B;
}

void numerics::logistic_regression::fit_linear(double lam) {
    arma::mat Phi = arma::join_rows(arma::ones(x.n_rows,1), x);
    
    auto f = [lam,&Phi,this](const arma::vec& cc) -> double {
        arma::mat p = Phi*arma::reshape(cc, Phi.n_cols, y.n_cols);
        p = softmax(p);
        return 0.5*lam*arma::dot(c,c) -arma::accu(p % y);
    };
    
    arma::vec cc = arma::zeros(Phi.n_cols*y.n_cols);
    nelder_mead fmin;
    fmin.minimize(f,cc);
    c = arma::reshape(cc, Phi.n_cols, y.n_cols);
}

void numerics::logistic_regression::fit_no_replace(const arma::mat& X, const arma::mat& Y, double lam) {
    int m = X.n_rows, n = X.n_cols;
    arma::mat Phi = arma::ones(m, 1+n+n_obs);
    Phi.cols(1,n) = X;
    Phi.cols(n+1,n+n_obs) = rbf(X);
    auto f = [lam,&Y,&Phi,this](const arma::vec& dd) -> double {
        arma::mat p = Phi * arma::reshape(dd, Phi.n_cols, y.n_cols);
        p = softmax(p);
        return 0.5*lam*arma::norm(dd,"fro") - arma::accu(Y % p);
    };
    auto grad_f = [lam,&Y,&Phi,this](const arma::vec& dd) -> arma::vec {
        arma::mat p = Phi * arma::reshape(dd, Phi.n_cols, y.n_cols);
        p = softmax(p);
        p = arma::vectorise(-Phi.t()*(Y-p)) + lam*dd;
        return p;
    };

    arma::vec dd = arma::zeros(Phi.n_cols*y.n_cols);
    lbfgs fmin;
    fmin.minimize(f,grad_f,dd);
    d = arma::reshape(dd, Phi.n_cols, y.n_cols);
    c = d.rows(0,n);
    d = d.rows(n+1,n+n_obs);
}

void numerics::logistic_regression::fit(const arma::mat& X, const arma::mat& Y) {
    if (X.n_rows != Y.n_rows) {
        std::cerr << "logistic_regression::fit() error : x.n_rows = " << X.n_rows << " != " << Y.n_rows << " = y.n_rows" << std::endl
                  << "fit not performed." << std::endl;
        return;
    }
    x = X;
    n_obs = x.n_rows;
    y = Y;


    if (std::isnan(lambda) || lambda < 0) { // lambda to be determined by cross validation
        uint num_folds = 10;
        if (X.n_rows / num_folds < 10) {
            num_folds = 5;
            if (X.n_rows / num_folds < 10) {
                num_folds = 3;
            }
        }
        k_folds split(x, y, num_folds);

        int N = 25;
        L = arma::logspace(-2,3,N);

        cv_scores = arma::zeros(N);

        #pragma omp parallel // run parameter sweep in parallel
        #pragma omp for
        for (int i=0; i < N; ++i) {
            int r = 0;
            double score = 0;
            for (uint j=0; j < num_folds; ++j) {
                if (beta > 0) fit_no_replace(split.train_set_X(j),split.train_set_Y(j),L(i));
                else fit_linear(L(i));
                double score_new = arma::accu( split.test_set_Y(j) % predict_probabilities(split.test_set_X(j)) );
                if (std::isnan(score_new)) continue;
                score += score_new;
                r++;
            }
            if (r != 0) score /= r;
            else score = 0;
            
            cv_scores(i) = score;
        }
        int indmax = cv_scores.index_max(); // using log-likelihood, so maximizing
        lambda = L(indmax);
    }

    if (beta > 0) fit_no_replace(x,y,lambda);
    else fit_linear(lambda);
}

arma::mat numerics::logistic_regression::predict_probabilities(const arma::mat& xgrid) {
    arma::mat p = arma::join_rows(arma::ones(xgrid.n_rows,1), xgrid) * c;
    if (beta > 0) p += rbf(xgrid)*d;
    return softmax(p);
}

arma::umat numerics::logistic_regression::predict_categories(const arma::mat& xgrid) {
    arma::uvec ind = arma::index_max(predict_probabilities(xgrid),1);
    arma::umat categories = arma::zeros<arma::umat>(ind.n_rows,c.n_cols);
    for (uint i=0; i < ind.n_rows; ++i) categories(i,ind(i)) = 1;
    return categories;
}

numerics::logistic_regression::logistic_regression(double Beta, double Lambda) {
    beta = Beta;
    lambda = Lambda;
}