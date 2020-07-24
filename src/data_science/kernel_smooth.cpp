#include <numerics.hpp>

void numerics::KernelSmooth::fit(const arma::mat& x, const arma::vec& y) {
    _check_xy(x,y);
    _dim = x.n_cols;
    if (x.n_cols > 1) {
        throw std::logic_error("KernelSmooth does not support 2+ dimensional data yet.");
    } else {
        if (_binning) _bins.fit(x, y);
        else {
            arma::uvec I = arma::sort_index(x);
            _X = x(I);
            _y = y(I);
        }
        double stddev = arma::stddev(x.as_col());
        if (_bdw <= 0) _bdw = bw::grid_mse(x, y, _kern, stddev, 40, _binning);
    }
}

arma::vec numerics::KernelSmooth::predict(const arma::mat& x) const {
    _check_x(x);
    if (_bdw <= 0) {
        throw std::runtime_error("KernelSmooth object not fitted.");
    }
    if (x.n_cols > 1) {
        throw std::logic_error("KernelSmooth does not support 2+ dimensional data yet.");
    } else {
        arma::vec yhat(x.n_elem);
        for (u_int i=0; i < x.n_elem; ++i) {
            arma::vec r;
            if (_binning) r = arma::abs(_bins.bins - x(i))/_bdw;
            else r = arma::abs(_X - x(i))/_bdw;

            arma::vec K = bw::eval_kernel(r, _kern);

            if (_binning) yhat(i) = arma::dot(_bins.counts, K) / arma::sum(K);
            else yhat(i) = arma::dot(y, K) / arma::sum(K); 
        }
        return yhat;
    }

}

double numerics::KernelSmooth::score(const arma::mat& x, const arma::vec& y) const {
    _check_xy(x,y);
    return r2_score(y, predict(x));
}

// /* kernel_smooth(k, estim, binning) : initialize kernel smoothing object by specifying bandwidth estimation method
//  * --- k : choice of kernel. options include : RBF, square, triangle, parabolic
//  * --- estim : method of selecting bandwidth.
//  * --- binning : whether to bin data or not. */
// numerics::kernel_smooth::kernel_smooth(kernels k, bool binning) : data_x(x), data_y(y), bandwidth(bdw) {
//     kern = k;
//     this->binning = binning;
//     bdw = 0;
// }

// /* kernel_smooth(h, k, binning) : initialize kernel smoothing object by specifying bandwidth.
//  * --- bdw : kernel bandwidth. default bdw=0.0; when fitting the default value tells the object to choose a bandwidth by k-fold cross validation.
//  * --- k : choice of kernel. options include : RBF, square, triangle, parabolic
//  * --- binning : whether to bin data or not */
// numerics::kernel_smooth::kernel_smooth(double h, kernels k, bool binning) : data_x(x), data_y(y), bandwidth(bdw) {
//     if (h <= 0) {
//         std::cerr << "kernel_smooth::kernel_smooth() error: invalid choice for bandwidth (must be > 0 but bdw recieved = " << h << ").\n";
//         bdw = 0;
//     } else {
//         bdw = h;
//     }
//     kern = k;
//     this->binning = binning;
// }

// /* kernel_smooth(in) : initialize kernel smoothing object from saved instance in file */
// numerics::kernel_smooth::kernel_smooth(const std::string& fname) : data_x(x), data_y(y), bandwidth(bdw) {
//     load(fname);
// }

// /* save(out) : save kernel smoothing object to output stream. */
// void numerics::kernel_smooth::save(const std::string& fname) {
//     std::ofstream out(fname);
//     out << n << " " << (int)kern << " " << bdw << " " << std::endl;
//     x.t().raw_print(out);
//     y.t().raw_print(out);
// }

// /* load(in) : load kernel smoothing object from input stream. */
// void numerics::kernel_smooth::load(const std::string& fname) {
//     std::ifstream in(fname);
//     int k;
//     in >> n >> k >> bdw;
//     if (k==0) kern = kernels::gaussian;
//     else if (k==1) kern = kernels::square;
//     else if (k==2) kern = kernels::triangle;
//     else kern = kernels::parabolic;
//     x = arma::zeros(n);
//     y = arma::zeros(n);
//     for (uint i=0; i < n; ++i) in >> x(i);
//     for (uint i=0; i < n; ++i) in >> y(i);
// }

// /* predict(t) : predict a single value of the function
//  * --- t : input to predict response of. */
// double numerics::kernel_smooth::predict(double t) {
//     arma::vec tt = {t};
//     return predict(tt)(0);
// }

// /* predict(t) : predict values of the function
//  * --- t : input to predict response of. */
// arma::vec numerics::kernel_smooth::predict(const arma::vec& t) {
//     if (bdw <= 0) {
//         std::cerr << "kernel_smooth::predict() error: bandwidth must be strictly greater than 0" << std::endl;
//         return 0;
//     }

//     arma::vec yhat(t.n_elem);
//     for (int i=0; i < t.n_elem; ++i) {
//         arma::vec r;
//         if (binning) r = (bins.bins - t(i))/bdw;
//         else r = arma::abs(x - t(i))/bdw;

//         arma::vec K = bw::eval_kernel(r, kern);

//         if (binning) yhat(i) = arma::dot(bins.counts, K) / arma::sum(K);
//         else yhat(i) = arma::dot(y, K) / arma::sum(K); 
//     }
//     return yhat;
// }

// /* fit(X, Y) : supply x and y values to fit object to.
//  * --- X : data vector of independent variable
//  * --- Y : data vector of dependent variable
//  * returns reference to 'this', so function object can be continuously called. */
// numerics::kernel_smooth& numerics::kernel_smooth::fit(const arma::vec& X, const arma::vec& Y) {
//     if (X.n_elem != Y.n_elem) {
//         std::cerr << "kernel_smooth::fit() error: input vectors must have the same length." << std::endl
//                   << "x.n_elem = " << X.n_elem << " != " << Y.n_elem << " = y.n_elem" << std::endl;
//         return *this;
//     }

//     if (X.n_elem <= 2) {
//         std::cerr << "kernel_smooth::fit() error: input vectors must have length > 2 (input vector length = " << x.n_elem << ").\n";
//         return *this;
//     }

//     arma::uvec I = arma::sort_index(X);
//     x = X(I);
//     double stddev = arma::stddev(x);
//     y = Y(I);
//     n = x.n_elem;
//     if (binning) bins.to_bins(x,y);

//     if (bdw <= 0) bdw = smooth_grid_mse(x, y, kern, stddev, 40, binning);

//     return *this;
// }

// /* fit_predict(X, Y) : fit object to data and predict on the same data.
//  * --- X : data vector of independent variable
//  * --- Y : data vector of dependent variable
//  * note: this is equivalent to calling this.fit(X,Y).predict(X) */
// arma::vec numerics::kernel_smooth::fit_predict(const arma::vec& X, const arma::vec& Y) {
//     return fit(X,Y).predict(X);
// }

// /* operator() :  same as predict(double t) */
// double numerics::kernel_smooth::operator()(double t) {
//     return predict(t);
// }

// /* operator() : same as predict(const arma::vec& t) */
// arma::vec numerics::kernel_smooth::operator()(const arma::vec& t) {
//     return predict(t);
// }