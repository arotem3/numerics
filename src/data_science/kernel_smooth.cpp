#include <numerics.hpp>

/* smooth(x, y, t, h, kern) : apply a smoothing kernel to 't' using data 'x' and 'y'.
 * --- x : training data, independent variable
 * --- y : training data, dependent variable
 * --- t : points to evaluate smoothing function over.
 * --- h : kernel bandwidth.
 * --- kern : smoothing kernel.
 * utility function for class kernel_smooth */
arma::vec smooth(const arma::vec& x, const arma::vec& y, const arma::vec& t, double h, numerics::kernels kern) {
    arma::vec yhat(t.n_elem);
    for (int i=0; i < t.n_elem; ++i) {
        arma::vec r;
        r = arma::abs(x - t(i))/h;

        arma::vec K = numerics::bw::eval_kernel(r, kern);

        yhat(i) = arma::dot(y, K) / arma::sum(K); 
    }
    return yhat;
}

/* smooth(xy, t, h, kern) : apply a smoothing kernel to 't' using data 'x' and 'y'.
 * --- xy : training data, pre-binned
 * --- t : points to evaluate smoothing function over.
 * --- h : kernel bandwidth.
 * --- kern : smoothing kernel.
 * utility function for class kernel_smooth */
arma::vec smooth(const numerics::bin_data& xy, const arma::vec& t, double h, numerics::kernels kern) {
    arma::vec yhat(t.n_elem);
    for (int i=0; i < t.n_elem; ++i) {
        arma::vec r;
        r = (xy.bins - t(i))/h;

        arma::vec K = numerics::bw::eval_kernel(r, kern);

        yhat(i) = arma::dot(xy.counts, K) / arma::sum(K);
    }
    return yhat;
}

/* smooth_grid_mse(x, y, kern, s=0, grid_size=20, binning=false) : compute an optimal bandwidth for kernel smoothing by grid search cross-validation using an approximate RMSE.
 * --- x : independent variable.
 * --- y : dependent variable.
 * --- kern : smoothing kernel.
 * --- s : precomputed standard deviation of x.
 * --- binning : whether to bin the data or not. */
double smooth_grid_mse(const arma::vec& x, const arma::vec& y, numerics::kernels kern, double s=0, int grid_size=20, bool binning=false) {
    if (s <= 0) s = arma::stddev(x);
    double max_h = (x(x.n_elem-1) - x(0))/4;
    double min_h = std::min(2*arma::median(arma::diff(x)), 0.05*s);

    arma::vec bdws = arma::logspace(std::log10(min_h), std::log10(max_h), grid_size);
    arma::vec cv_scores(grid_size);

    numerics::k_folds train_test(x,y);
    int n;
    if (x.n_elem > 1000) n = 500; // almost always sufficient
    else if (x.n_elem > 30) n = x.n_elem / 10; // somewhat arbitrary... we want _n << x.n_elem
    else if (x.n_elem/5 > 5) n = x.n_elem / 5;
    else n = x.n_elem;
    numerics::bin_data bins(n);
    if (binning) bins.to_bins(train_test.train_set_X(0), train_test.train_set_Y(0));

    for (int i=0; i < grid_size; ++i) {
        arma::vec yhat;
        if (binning) yhat = smooth(bins, train_test.test_set_X(0), bdws(i), kern);
        else yhat = smooth(train_test.train_set_X(0), train_test.train_set_Y(0), train_test.test_set_X(0), bdws(i), kern);
        cv_scores(i) = arma::norm(yhat - train_test.test_set_Y(0));
    }

    int imin = cv_scores.index_min();
    return bdws(imin);
}

/* kernel_smooth(k, estim, binning) : initialize kernel smoothing object by specifying bandwidth estimation method
 * --- k : choice of kernel. options include : RBF, square, triangle, parabolic
 * --- estim : method of selecting bandwidth.
 * --- binning : whether to bin data or not. */
numerics::kernel_smooth::kernel_smooth(kernels k, bool binning) : data_x(x), data_y(y), bandwidth(bdw) {
    kern = k;
    this->binning = binning;
    bdw = 0;
}

/* kernel_smooth(h, k, binning) : initialize kernel smoothing object by specifying bandwidth.
 * --- bdw : kernel bandwidth. default bdw=0.0; when fitting the default value tells the object to choose a bandwidth by k-fold cross validation.
 * --- k : choice of kernel. options include : RBF, square, triangle, parabolic
 * --- binning : whether to bin data or not */
numerics::kernel_smooth::kernel_smooth(double h, kernels k, bool binning) : data_x(x), data_y(y), bandwidth(bdw) {
    if (h <= 0) {
        std::cerr << "kernel_smooth::kernel_smooth() error: invalid choice for bandwidth (must be > 0 but bdw recieved = " << h << ").\n";
        bdw = 0;
    } else {
        bdw = h;
    }
    kern = k;
    this->binning = binning;
}

/* kernel_smooth(in) : initialize kernel smoothing object from saved instance in file */
numerics::kernel_smooth::kernel_smooth(const std::string& fname) : data_x(x), data_y(y), bandwidth(bdw) {
    load(fname);
}

/* save(out) : save kernel smoothing object to output stream. */
void numerics::kernel_smooth::save(const std::string& fname) {
    std::ofstream out(fname);
    out << n << " " << (int)kern << " " << bdw << " " << std::endl;
    x.t().raw_print(out);
    y.t().raw_print(out);
}

/* load(in) : load kernel smoothing object from input stream. */
void numerics::kernel_smooth::load(const std::string& fname) {
    std::ifstream in(fname);
    int k;
    in >> n >> k >> bdw;
    if (k==0) kern = kernels::gaussian;
    else if (k==1) kern = kernels::square;
    else if (k==2) kern = kernels::triangle;
    else kern = kernels::parabolic;
    x = arma::zeros(n);
    y = arma::zeros(n);
    for (uint i=0; i < n; ++i) in >> x(i);
    for (uint i=0; i < n; ++i) in >> y(i);
}

/* predict(t) : predict a single value of the function
 * --- t : input to predict response of. */
double numerics::kernel_smooth::predict(double t) {
    arma::vec tt = {t};
    return predict(tt)(0);
}

/* predict(t) : predict values of the function
 * --- t : input to predict response of. */
arma::vec numerics::kernel_smooth::predict(const arma::vec& t) {
    if (bdw <= 0) {
        std::cerr << "kernel_smooth::predict() error: bandwidth must be strictly greater than 0" << std::endl;
        return 0;
    }

    arma::vec yhat(t.n_elem);
    for (int i=0; i < t.n_elem; ++i) {
        arma::vec r;
        if (binning) r = (bins.bins - t(i))/bdw;
        else r = arma::abs(x - t(i))/bdw;

        arma::vec K = bw::eval_kernel(r, kern);

        if (binning) yhat(i) = arma::dot(bins.counts, K) / arma::sum(K);
        else yhat(i) = arma::dot(y, K) / arma::sum(K); 
    }
    return yhat;
}

/* fit(X, Y) : supply x and y values to fit object to.
 * --- X : data vector of independent variable
 * --- Y : data vector of dependent variable
 * returns reference to 'this', so function object can be continuously called. */
numerics::kernel_smooth& numerics::kernel_smooth::fit(const arma::vec& X, const arma::vec& Y) {
    if (X.n_elem != Y.n_elem) {
        std::cerr << "kernel_smooth::fit() error: input vectors must have the same length." << std::endl
                  << "x.n_elem = " << X.n_elem << " != " << Y.n_elem << " = y.n_elem" << std::endl;
        return *this;
    }

    if (X.n_elem <= 2) {
        std::cerr << "kernel_smooth::fit() error: input vectors must have length > 2 (input vector length = " << x.n_elem << ").\n";
        return *this;
    }

    arma::uvec I = arma::sort_index(X);
    x = X(I);
    double stddev = arma::stddev(x);
    y = Y(I);
    n = x.n_elem;
    if (binning) bins.to_bins(x,y);

    if (bdw <= 0) bdw = smooth_grid_mse(x, y, kern, stddev, 40, binning);

    return *this;
}

/* fit_predict(X, Y) : fit object to data and predict on the same data.
 * --- X : data vector of independent variable
 * --- Y : data vector of dependent variable
 * note: this is equivalent to calling this.fit(X,Y).predict(X) */
arma::vec numerics::kernel_smooth::fit_predict(const arma::vec& X, const arma::vec& Y) {
    return fit(X,Y).predict(X);
}

/* operator() :  same as predict(double t) */
double numerics::kernel_smooth::operator()(double t) {
    return predict(t);
}

/* operator() : same as predict(const arma::vec& t) */
arma::vec numerics::kernel_smooth::operator()(const arma::vec& t) {
    return predict(t);
}