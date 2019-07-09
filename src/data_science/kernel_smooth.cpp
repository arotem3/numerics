#include <numerics.hpp>

/* kernel_smooth(bdw, k) : initialize kernel smoothing object
 * --- bdw : kernel bandwidth. default bdw=0.0; when fitting the default value tells the object to choose a bandwidth by k-fold cross validation.
 * --- k : choice of kernel. options include : RBF, square, triangle, parabolic */
numerics::kernel_smooth::kernel_smooth(double bdw, kernels k) {
    kern = k;
    this->bdw = bdw;
}

/* kernel_smooth(X, Y, bdw, k) : initialize and fit kernel smoothing object.
 * --- X : data vector of independent variable
 * --- Y : data vector of dependent variable
 * --- bdw : kernel bandwidth. default bdw=0.0; when fitting the default value tells the object to choose a bandwidth by k-fold cross validation.
 * --- k : choice of kernel. options include : RBF, square, triangle, parabolic */
numerics::kernel_smooth::kernel_smooth(const arma::vec& X, const arma::vec& Y, double bdw, kernels k) {
    if (X.n_elem != Y.n_elem) {
        std::cerr << "kernel_smooth() error: input vectors must have the same length." << std::endl
                  << "x.n_elem = " << X.n_elem << " != " << Y.n_elem << " = y.n_elem" << std::endl;
        return;
    }
    
    kern = k;
    x = X;
    y = Y;
    n = x.n_elem;
    int num_folds = 10;
    if (n/num_folds < 10) { // choose ideal num_folds.
        num_folds = 5;
        if (n/num_folds < 10) {
            num_folds = 3;
        }
    }
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
        
        #pragma omp parallel
        #pragma omp for
        for (int i=0; i < m; ++i) {
            double score = 0;
            for (int j=0; j < num_folds; ++j) {
                ind = arma::find(arma::regspace(0,num_folds-1) != j);
                arma::umat ii = I.rows(ind);
                ii = arma::vectorise(ii);
                arma::vec s = predict(x(ii), y(ii), x(I.row(j).t()), bdws(i)) - y(I.row(j).t());
                score += arma::dot(s,s);
            }
            #pragma omp critical
            cv_score(i) += score;
        }
        cv_score /= num_folds;
        int imin = cv_score.index_min();
        this->bdw = bdws(imin);
        this->cv = cv_score(imin);
    }
}

/* kernel_smooth(in) : initialize kernel smoothing object from saved instance in file */
numerics::kernel_smooth::kernel_smooth(std::istream& in) {
    load(in);
}

/* save(out) : save kernel smoothing object to output stream. */
void numerics::kernel_smooth::save(std::ostream& out) {
    out << n << " " << (int)kern << " " << bdw << " " << cv << std::endl;
    x.t().raw_print(out);
    y.t().raw_print(out);
}

/* load(in) : load kernel smoothing object from input stream. */
void numerics::kernel_smooth::load(std::istream& in) {
    int k;
    in >> n >> k >> bdw >> cv;
    if (k==0) kern = RBF;
    else if (k==1) kern = square;
    else if (k==2) kern = triangle;
    else kern = parabolic;
    x = arma::zeros(n);
    y = arma::zeros(n);
    for (uint i=0; i < n; ++i) in >> x(i);
    for (uint i=0; i < n; ++i) in >> y(i);
}

/* predict(X, Y, t, h) : *private* predicts according to input bandwidth. */
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

/* predict(t) : predict a single value of the function
 * --- t : input to predict response of. */
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

/* predict(t) : predict values of the function
 * --- t : input to predict response of. */
arma::vec numerics::kernel_smooth::predict(const arma::vec& t) {
    arma::vec yhat = arma::zeros(arma::size(t));
    for (uint i=0; i < t.n_elem; ++i) {
        yhat(i) = predict(t(i));
    }
    return yhat;
}

/* fit(X, Y) : supply x and y values to fit object to.
 * --- X : data vector of independent variable
 * --- Y : data vector of dependent variable
 * returns reference to 'this', so function object can be continuously called. */
numerics::kernel_smooth& numerics::kernel_smooth::fit(const arma::vec& X, const arma::vec& Y) {
    kernel_smooth(X,Y);
    return *this;
}

/* fit_predict(X, Y) : fit object to data and predict on the same data.
 * --- X : data vector of independent variable
 * --- Y : data vector of dependent variable
 * note: this is equivalent to calling this.fit(X,Y).predict(X) */
arma::vec numerics::kernel_smooth::fit_predict(const arma::vec& X, const arma::vec& Y) {
    return fit(X,Y).predict(X);
}

/* MSE() : return MSE of fit produced from cross validation on data set. */
double numerics::kernel_smooth::MSE() const {
    return cv;
}

/* operator() :  same as predict(double t) */
double numerics::kernel_smooth::operator()(double t) {
    return predict(t);
}

/* operator() : same as predict(const arma::vec& t) */
arma::vec numerics::kernel_smooth::operator()(const arma::vec& t) {
    return predict(t);
}

/* data_X : returns the independent variable data provided durring call to fit or initialization. */
arma::vec numerics::kernel_smooth::data_X() {
    return x;
}

/* data_Y : returns the dependent variable data provided durring call to fit or initialization. */
arma::vec numerics::kernel_smooth::data_Y() {
    return y;
}

/* bandwidth : returns either the bandwith provided durring initialization or the bandwidth computed from cross validation. */
double numerics::kernel_smooth::bandwidth() const {
    return bdw;
}   