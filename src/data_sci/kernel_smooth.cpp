#include "numerics.hpp"

numerics::kernel_smooth::kernel_smooth(double bdw, kernels k) {
    kern = k;
    this->bdw = bdw;
}

numerics::kernel_smooth::kernel_smooth(const arma::vec& X, const arma::vec& Y, double bdw, kernels k) {
    if (X.n_elem != Y.n_elem) {
        std::cerr << "kernel_smooth() error: input vectors must have the same length." << std::endl
                  << "x.n_elem = " << X.n_elem << " != " << Y.n_elem << " = y.n_elem" << std::endl;
        return;
    }
    
    kern = k;
    int num_folds = 10;
    x = X;
    y = Y;
    n = x.n_elem;
    arma::uvec ind = arma::regspace<arma::uvec>(0,n-1);
    arma::umat I = arma::shuffle(ind);
    I = arma::reshape(I, num_folds, n/num_folds);

    if (bdw != 0) {
        this->bdw = bdw;
        cv = 0;
        for (int j=0; j < num_folds; ++j) {
            ind = arma::find(arma::regspace(0,num_folds-1) != j);
            arma::umat ii = I.rows(ind);
            ii = arma::vectorise(ii);
            arma::vec s = predict(x(ii), y(ii), x(I.row(j)), bdw) - y(I.row(j));
            cv += arma::dot(s,s);
        }
        cv /= num_folds;
    } else {
        double max_interval = arma::range(x)/2;
        double min_interval = 2*arma::min(arma::diff(arma::sort(x)));
        min_interval = std::max(min_interval, 1e-2);

        int m = 200;
        arma::vec bdws = arma::logspace(std::log10(min_interval), std::log10(max_interval), m);
        arma::vec cv_score = arma::zeros(m);
        
        for (int i=0; i < m; ++i) {
            for (int j=0; j < num_folds; ++j) {
                ind = arma::find(arma::regspace(0,num_folds-1) != j);
                arma::umat ii = I.rows(ind);
                ii = arma::vectorise(ii);
                arma::vec s = predict(x(ii), y(ii), x(I.row(j).t()), bdws(i)) - y(I.row(j).t());
                cv_score(i) += arma::dot(s,s);
            }
        }
        cv_score /= num_folds;
        int imin = cv_score.index_min();
        this->bdw = bdws(imin);
        this->cv = cv_score(imin);
    }
}

arma::vec numerics::kernel_smooth::predict(const arma::vec& X, const arma::vec& Y, const arma::vec& t, double h) {
    arma::vec yhat = arma::zeros(arma::size(t));
    for (int i=0; i < yhat.n_elem; ++i) {
        arma::vec r = arma::abs(X - t(i))/h;
        arma::vec K;
        if (kern == kernels::RBF) {
            K = arma::exp(-0.5*arma::pow(r,2))/std::sqrt(2*M_PI);
        } else if (kern == kernels::square) {
            K = arma::zeros(arma::size(r));
            K(arma::find(r <= 1)) += 0.5;
        } else if (kern == kernels::triangle) {
            K = 1 - r;
            K(arma::find(K < 0)) *= 0;
        } else { // kernels::parabolic
            K = 0.75 * (1 - r%r);
            K(arma::find(K < 0)) *= 0;
        }
        yhat(i) = arma::dot(K,Y)/arma::sum(K);
    }
    return yhat;
}

double numerics::kernel_smooth::predict(double t) {
    if (bdw == 0) {
        std::cerr << "kernel_smooth::predict() error: bandwidth must be strictly greater than 0" << std::endl;
        return 0;
    }
    arma::vec r = arma::abs(x - t)/bdw;
    arma::vec K;
    if (kern == kernels::RBF) {
        K = arma::exp(-0.5*arma::pow(r,2))/std::sqrt(2*M_PI);
    } else if (kern == kernels::square) {
        K = arma::zeros(arma::size(r));
        K(arma::find(r <= 1)) += 0.5;
    } else if (kern == kernels::triangle) {
        K = 1 - r;
        K(arma::find(K < 0)) *= 0;
    } else { // kernels::parabolic
        K = 0.75 * (1 - r%r);
        K(arma::find(K < 0)) *= 0;
    }
    return arma::dot(K,y)/arma::sum(K);
}

arma::vec numerics::kernel_smooth::predict(const arma::vec& t) {
    arma::vec yhat = arma::zeros(arma::size(t));
    for (uint i=0; i < t.n_elem; ++i) {
        yhat(i) = predict(t(i));
    }
    return yhat;
}

void numerics::kernel_smooth::fit(const arma::vec& X, const arma::vec& Y) {
    kernel_smooth(X,Y);
}

double numerics::kernel_smooth::CV_score() const {
    return cv;
}

double numerics::kernel_smooth::operator()(double t) {
    return predict(t);
}

arma::vec numerics::kernel_smooth::operator()(const arma::vec& t) {
    return predict(t);
}

arma::vec numerics::kernel_smooth::data_X() {
    return x;
}

arma::vec numerics::kernel_smooth::data_Y() {
    return y;
}

double numerics::kernel_smooth::bandwidth() {
    return bdw;
}