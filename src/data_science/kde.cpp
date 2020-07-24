#include<numerics.hpp>

void numerics::KDE::fit(const arma::mat& x) {
    _dim = x.n_cols;
    if (_dim > 1) {
        throw std::logic_error("KDE does not support 2+ dimensional data yet.");
    } else {        
        if (_binning) {
            _bins.fit(x);
            if (_bdw <= 0) {
                if (_bandwidth_estimator == "rule_of_thumb") _bdw = bw::rot1(_bins.n_bins, arma::stddev(x.as_col()));
                else if (_bandwidth_estimator == "min_sd_iqr") _bdw = bw::rot2(x);
                else if (_bandwidth_estimator == "plug_in") _bdw = bw::dpi_binned(_bins, 0, _kern);
                else if (_bandwidth_estimator == "grid_cv") _bdw = bw::grid_mse(x, _kern, 0, 40, _binning);
            }
        } else {
            _X = x;
            if (_bdw <= 0) {
                if (_bandwidth_estimator == "rule_of_thumb") _bdw = bw::rot1(x.n_elem, arma::stddev(x.as_col()));
                else if (_bandwidth_estimator == "min_sd_iqr") _bdw = bw::rot2(x);
                else if (_bandwidth_estimator == "plug_in") _bdw = bw::dpi(x, 0, _kern);
                else if (_bandwidth_estimator == "grid_cv") _bdw = bw::grid_mse(x, _kern, 0, 40, _binning);
            }
        }
    }
}

arma::vec numerics::KDE::fit_predict(const arma::mat& x) {
    fit(x);
    return predict(x);
}

arma::vec numerics::KDE::predict(const arma::mat& x) const {
    _check_x(x);
    if (_bdw <= 0) {
        throw std::runtime_error("KDE object was never fitted.");
    }
    arma::vec fhat(x.n_rows);
    if (_dim > 1) {
        throw std::logic_error("KDE does not support 2+ dimensional data yet.");
    } else {
        for (u_int i=0; i < fhat.n_rows; ++i) {
            arma::vec r;
            if (_binning) r = arma::abs(_bins.bins - x(i))/_bdw;
            else r = arma::abs(_X - x(i))/_bdw;

            arma::vec K = bw::eval_kernel(r, _kern);

            if (_binning) fhat(i) = arma::dot(_bins.counts, K) / (arma::accu(_bins.counts) * _bdw);
            else fhat(i) = arma::sum(K) / (_X.n_rows * _bdw);
        }
    }
    return fhat;
}

arma::vec numerics::KDE::sample(int N) const {
    if (_bdw <= 0) {
        throw std::runtime_error("KDE object was never fitted.");
    }
    arma::mat rvals;
    if (_dim > 1) {
        throw std::runtime_error("KDE does not support 2+ dimensional data yet.");
    } else {
        if (_binning) rvals = _bins.bins(sample_from(N, _bins.counts/arma::accu(_bins.counts)));
        else rvals = _X( arma::randperm(_X.n_rows, N) );

        arma::vec kern_rvals;
        if (_kern == "gaussian") {
            kern_rvals = arma::randn(N);
        } else if (_kern == "square") {
            kern_rvals = arma::randu(N)-0.5;
        } else if (_kern == "triangle") {
            kern_rvals = arma::randu(N);
            arma::uvec less_half = arma::find(kern_rvals < 0.5);
            arma::uvec greater_half = arma::find(kern_rvals >= 0.5);
            kern_rvals(less_half) = -1 * arma::sqrt(2 * kern_rvals(less_half));
            kern_rvals(greater_half) = 1 - arma::sqrt(2 - 2*kern_rvals(greater_half));
        } else if (_kern == "parabolic") {
            kern_rvals = arma::randu(N);
            kern_rvals = 2*arma::sin(arma::asin(2*kern_rvals-1)/3); // inverse cdf for parabolic kernel
        }
        kern_rvals *= _bdw;
        rvals += kern_rvals;
    }
    return rvals;
}

// /* kde(k=gaussian, estim=min_sd_iqr, bin=false) : initialize kde object without specifying a bandwidth.
//  * --- k : kernel to use.
//  * --- estim : method for estimating the kernel bandwidth.
//  * --- bin : whether to bin the data or not. */
// numerics::kde::kde(kernels k, bandwidth_estimator estim, bool bin) : bandwidth(bdw), data(x) {
//     kern = k;
//     method = estim;
//     binning = bin;
//     bdw = 0;
// }

// /* kde(h, k, bin) : initialize kde object by specifying a bandwidth.
//  * --- h : bandwidth to use.
//  * --- k : kernel to use.
//  * --- bin : whether to bin the data or not. */
// numerics::kde::kde(double h, kernels k, bool bin) : bandwidth(bdw), data(x) {
//     if (h <= 0) {
//         std::cerr << "kde::kde() error: invalid choice for bandwidth (must be > 0 but bdw recieved = " << h << ").\n";
//         bdw = 0;
//         method = bandwidth_estimator::min_sd_iqr;
//     } else {
//         bdw = h;
//     }
//     binning = bin;
// }

// /* kde(fname) : initialize kde object by loading in previously constructed and saved object. */
// numerics::kde::kde(const std::string& fname) : bandwidth(bdw), data(x) {
//     load(fname);
// }

// /* fit(data) : fit kernel density estimator.
//  * data : data to fit kde over. */
// numerics::kde& numerics::kde::fit(const arma::vec& data) {
//     if (binning) bins.to_bins(data);
//     x = arma::sort(data);
//     stddev = arma::stddev(x);

//     if (bdw <= 0) {
//         if (method == bandwidth_estimator::rule_of_thumb_sd) {
//             if (binning) bdw = bw::rot1(bins.n_bins, stddev);
//             else bdw = bw::rot1(x.n_elem, stddev);
//         } else if (method == bandwidth_estimator::min_sd_iqr) {
//             if (binning) bdw = bw::rot2(x, bins.n_bins);
//             else bdw = bw::rot2(x, stddev);
//         } else if (method == bandwidth_estimator::direct_plug_in) {
//             if (binning) bdw = bw::dpi_binned(bins, stddev, kern);
//             else bdw = bw::dpi(x, stddev, kern);
//         } else bdw = bw::grid_mse(x, kern, 0, 40, binning); // method == bandwidth::grid_cv
//     }
//     return *this;
// }

// /* sample(n=1) : produce random sample from density estimate.
//  * --- n : sample size. */
// arma::vec numerics::kde::sample(uint N) {
//     arma::vec rvals;

//     if (binning) rvals = bins.bins(sample_from(N, bins.counts/x.n_elem));
//     else rvals = x( arma::randi<arma::uvec>(N, arma::distr_param(0,x.n_elem-1)) );

//     arma::vec kern_rvals;
//     if (kern == kernels::gaussian) {
//         kern_rvals = arma::randn(N);
//     } else if (kern == kernels::square) {
//         kern_rvals = (arma::randu(N)-0.5);
//     } else if (kern == kernels::triangle) {
//         kern_rvals = arma::randu(N);
//         arma::uvec less_half = arma::find(kern_rvals < 0.5);
//         arma::uvec greater_half = arma::find(kern_rvals >= 0.5);
//         kern_rvals(less_half) = -1 * arma::sqrt(2 * kern_rvals(less_half));
//         kern_rvals(greater_half) = 1 - arma::sqrt(2 - 2*kern_rvals(greater_half));
//     } else { // kern == kernels::parabolic
//         kern_rvals = arma::randu(N);
//         kern_rvals = 2*arma::sin(arma::asin(2*kern_rvals-1)/3); // inverse cdf for parabolic kernel
//     }
//     kern_rvals *= bdw;
//     rvals += kern_rvals;
//     return rvals;
// }

// /* predict(t) : predict density for new data.
//  * --- t : query point. */
// double numerics::kde::predict(double t) {
//     arma::vec tt = {t};
//     return predict(tt)(0);
// }

// /* operator()(t) : same as predict(t).
//  * --- t : query point. */
// double numerics::kde::operator()(double t) {
//     return predict(t);
// }

// /* predict(t) : predict density for new data.
//  * --- t : vector of query points. */
// arma::vec numerics::kde::predict(const arma::vec& t) {
//     if (bdw <= 0) {
//         std::cerr << "kde::predict() error: the estimator was never fitted; returning empty vector.\n";
//         return arma::vec();
//     }
//     arma::vec fhat = arma::zeros(arma::size(t));
//     for (int i=0; i < fhat.n_elem; ++i) {
//         arma::vec r;
//         if (binning) r = arma::abs(bins.bins - t(i))/bdw;
//         else r = arma::abs(x - t(i))/bdw;

//         arma::vec K = bw::eval_kernel(r, kern);

//         if (binning) fhat(i) = arma::dot(bins.counts, K) / (x.n_elem * bdw);
//         else fhat(i) = arma::sum(K) / (x.n_elem * bdw);
//     }
//     return fhat;
// }

// /* operator()(t) : predict density for new data.
//  * --- t : vector of query points. */
// arma::vec numerics::kde::operator()(const arma::vec& t) {
//     return predict(t);
// }

// /* save(fname="kde.kde") : save kde to file. */
// void numerics::kde::save(const std::string& fname) {
//     std::ofstream out(fname);
//     out << x.n_elem << " " << bdw << " " << stddev << " " << (int)binning << " " << (int)kern << "\n";
//     x.t().raw_print(out);
// }

// /* load(fname) : load kde object from file. */
// void numerics::kde::load(const std::string& fname) {
//     std::ifstream in(fname);
//     int a,b,c;
//     in >> a >> bdw >> stddev >> b >> c;
//     if (a < 0 || b < 0 || c < 0 || c > 3) {
//         std::cerr << "kde::load error: invalid entry in file resulted in premature termination of load().\n";
//         return;
//     }
//     x = arma::vec(a);
//     binning = (bool)b;
//     kern = (kernels)c;
//     for (int i=0; i < a; ++i) in >> x(i);
//     if (binning) bins.to_bins(x);
// }