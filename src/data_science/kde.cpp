#include<numerics.hpp>

/* integral_R(K) : compute the integral of K^2 over [-inf, inf]. R(K) is the notation in the literature.
 * --- K : Kernel name.
 * utility for dpi */
inline double integral_R(numerics::kernels K) {
    if (K == numerics::kernels::gaussian) return 1/std::sqrt(4*M_PI);
    else if (K == numerics::kernels::parabolic) return 0.6;
    else if (K == numerics::kernels::square) return 0.5;
    else return 2.0/3; // triangular
}

/* second_moment(K) : compute the second moment of K.
 * --- K : kernel name.
 * utility for dpi */
inline double second_moment(numerics::kernels K) {
    if (K == numerics::kernels::gaussian) return 1.0;
    else if (K == numerics::kernels::parabolic) return 0.2;
    else if (K == numerics::kernels::square) return 1.0/3;
    else return 1.0/6; // triangular
}

/* dpi_Psi6 : utility for bdw_dpi */
inline double dpi_Psi6(const arma::mat& pairwise_diffs, double g6) {
    arma::mat K = pairwise_diffs / g6;
    K = arma::exp(-K%K/2) * (arma::pow(K,6) - 15*arma::pow(K,4) + 45*arma::pow(K,2) - 15);
    double Psi6 = arma::accu(K) / std::sqrt(2*M_PI);
    Psi6 /= std::pow(g6,7) * K.n_elem;
    return std::abs(Psi6);
}

/* dpi_Psi4 : utility for bdw_dpi */
inline double dpi_Psi4(const arma::mat& pairwise_diffs, double g4) {
    arma::mat K = pairwise_diffs / g4;
    K = arma::exp(-K%K/2) * (arma::pow(K,4) - 6*arma::pow(K,2) + 3);
    double Psi4 = arma::accu(K) / std::sqrt(2*M_PI);
    Psi4 /= std::pow(g4, 5) * K.n_elem;
    return std::abs(Psi4);
}

/* dpi(x, s=0, K=gaussian) : compute the direct plug in (L=2) bandwidth estimate for kernel density estimation.
 * --- x : data to estimate bandwidth for.
 * --- s : precomputed standard deviation of x, if s <= 0, then s will be computed.
 * --- K : kernel. */
double numerics::bw::dpi(const arma::vec& x, double s, numerics::kernels K) {
    if (s <= 0) s = arma::stddev(x);
    arma::mat pairwise_diff = arma::repmat(x, 1, x.n_elem);
    pairwise_diff = pairwise_diff - pairwise_diff.t();
    double Psi8 = 105.0 / (32.0 * std::sqrt(M_PI) * std::pow(s, 9));
    double g6 = std::pow( ( 2*15/std::sqrt(2*M_PI) ) / (Psi8*x.n_elem) , 1.0/9L );
    double Psi6 = dpi_Psi6(pairwise_diff, g6);
    double g4 = std::pow( (2*3/std::sqrt(2*M_PI)) / (Psi6*x.n_elem) , 1.0/7L );
    double Psi4 = dpi_Psi4(pairwise_diff, 4);
    double h_dpi = std::pow( integral_R(K) / (std::pow(second_moment(K),2) * Psi4 * x.n_elem) , 0.2L );
    return h_dpi;
}

/* dpi_binned(bins, s=0,  K=gaussian) : compute the direct plug in (L=2) bandwidth estimate for kernel density estimation for pre-binned data.
 * --- bins : bin_data object of prebinned data.
 * --- s : precomputed standard deviation of data, if s <= 0, then s will be computed.
 * --- K : kernel. */
double numerics::bw::dpi_binned(const numerics::bin_data& bins, double s, numerics::kernels K) {
    int n = bins.n_bins;
    if (s <= 0) {
        double m = arma::dot(bins.bins, bins.counts);
        double s = arma::norm( (m - bins.bins)%bins.counts );
    }
    double h = bins.bin_width;
    double Psi8 = 105.0 / (32.0 * std::sqrt(M_PI) * std::pow(s, 9));
    double g6 = std::pow( ( 2*15/std::sqrt(2*M_PI) ) / (Psi8*n) , 1.0/9L );
    double Psi6 = std::exp(-h*h/2) * (std::pow(h,6) - 15*std::pow(h,4) + 45*std::pow(h,2) - 15);
    Psi6 /= std::sqrt(2*M_PI) * std::pow(g6,7);
    Psi6 = std::abs(Psi6);
    double g4 = std::pow( (2*3/std::sqrt(2*M_PI)) / (Psi6*n) , 1.0/7L );
    double Psi4 = std::exp(-h*h/2) * (std::pow(h,4) - 6*h*h + 3);
    Psi4 /= std::pow(g4,5);
    Psi4 = std::abs(Psi4);
    double h_dpi = std::pow( integral_R(K) / (std::pow(second_moment(K),2) * Psi4 * n) , 0.2L );
    return h_dpi;
}

/* rot1(n, s) : the original rule of thumb bdw = 1.06 * s * n^(-1/5). Originally proposed by Silverman (1986) is optimal whenever the true distribution is normal.
 * --- n : size of data.
 * --- s : standard deviation of data. */
double numerics::bw::rot1(int n, double s) {
    return 1.06*s*std::pow(n, -0.2);
}

/* rot2(x, s=0) : a more common rule of thumb bdw = 0.9 * min(IQR/1.34, s) * n^(-1/5). More robust at treating mixture models and skew distributions.
 * --- x : data to compute bandwidth estimate for.
 * --- s : precomputed standard deviation of data, if s <= 0, then s will be computed. */
double numerics::bw::rot2(const arma::vec& x, double s) {
    if (s <= 0) s = arma::stddev(x);
    double q1,q2,q3;
    q2 = arma::median(x);
    q1 = arma::median( x(arma::find(x < q2)) );
    q3 = arma::median( x(arma::find(x >= q2)) );
    double iqr = q3 - q1;
    return 0.9 * std::min(iqr/1.34, s) * std::pow(x.n_elem, -0.2);
}

arma::vec numerics::bw::eval_kernel(const arma::vec& r, numerics::kernels kern) {
    arma::vec K;
    if (kern == kernels::gaussian) {
        K = arma::exp(-0.5*arma::pow(r,2))/std::sqrt(2*M_PI);
    } else if (kern == kernels::square) {
        K = arma::zeros(arma::size(r));
        K(arma::find(r <= 1)) += 0.5;
    } else if (kern == kernels::triangle) {
        K = 1 - r;
        K(arma::find(K < 0)) *= 0;
    } else { // kernels::parabolic
        K = 0.75 * (1 - arma::pow(r,2));
        K(arma::find(K < 0)) *= 0;
    }
    return K;
}

/* density(x, n, t, h, kern) : approximate density using kernel density estimation over binned data. (utility function for class kde)
 * --- x : bin_data object of prebinned data.
 * --- n : size of data.
 * --- t : points to evaluate density for.
 * --- h : bandwidth.
 * --- kern : kernel. */
arma::vec density(const numerics::bin_data& x, int n, const arma::vec& t, double h, numerics::kernels kern) {
    arma::vec fhat = arma::zeros(arma::size(t));
    for (int i=0; i < fhat.n_elem; ++i) {
        arma::vec r = arma::abs(x.bins - t(i))/h;

        arma::vec K = numerics::bw::eval_kernel(r, kern);

        fhat(i) = arma::dot(x.counts, K) / (n * h);
    }
    return fhat;
}

/* denisty(x, t, h, kern) : approximate density using kernel density estimation. (utility function for class kde)
 * --- x : data.
 * --- t : points to evaluate density over.
 * --- h : bandwidth.
 * --- kern : kernel. */
arma::vec density(const arma::vec& x, const arma::vec& t, double h, numerics::kernels kern) {
    arma::vec fhat = arma::zeros(arma::size(t));
    for (int i=0; i < fhat.n_elem; ++i) {
        arma::vec r = arma::abs(x - t(i))/h;

        arma::vec K = numerics::bw::eval_kernel(r, kern);

        fhat(i) = arma::sum(K) / (x.n_elem * h);
    }
    return fhat;
}

/* bdw_grid_mse(x, K, s=0, grid_size=20, binning=false) : compute an optimal bandwidth for kernel density estimation by grid search cross-validation using an approximate RMSE, where the true density is estimated using a pilot density. The pilot density is computed using the entire data set and the bdw_rot2 estimate for the bandwidth. Then the MSE for each bandwidth is computed by predicting the density for a testing subset of the data computed using the rest of the data.
 * --- x : data to estimate bandwidth for.
 * --- K : kernel to use.
 * --- s : precomputed standard deviation.
 * --- grid_size : number of bandwidths to test, the range being [0.05*s, range(x)/4] using log spacing.
 * --- binning : whether to prebin the data. Typically the estimate is just as good, but the computational cost is significantly reduced. */
double numerics::bw::grid_mse(const arma::vec& x, numerics::kernels K, double s, int grid_size, bool binning) {
    if (s <= 0) s = arma::stddev(x);
    double max_h = (x(x.n_elem-1) - x(0))/4;
    double min_h = std::min(2*arma::median(arma::diff(x)), 0.05*s);

    arma::vec bdws = arma::logspace(std::log10(min_h), std::log10(max_h), grid_size);
    arma::vec cv_scores(grid_size);

    numerics::k_folds_1d train_test(x);
    numerics::bin_data bins;
    if (binning) bins.to_bins(x);

    arma::vec pilot_density;
    if (binning) {
        pilot_density = density(bins, x.n_elem, train_test.test_set(0), bw::rot2(x), K);
        bins.to_bins(train_test.train_set(0));
    } else pilot_density = density(x, train_test.test_set(0), bw::rot2(x), K);

    for (int i=0; i < grid_size; ++i) {
        arma::vec pHat;
        if (binning) pHat = density(bins, x.n_elem, train_test.test_set(0), bdws(i), K);
        else pHat = density(train_test.train_set(0), train_test.test_set(0), bdws(i), K);
        cv_scores(i) = arma::norm(pHat - pilot_density);
    }

    int imin = cv_scores.index_min();
    return bdws(imin);
}

/* kde(k=gaussian, estim=min_sd_iqr, bin=false) : initialize kde object without specifying a bandwidth.
 * --- k : kernel to use.
 * --- estim : method for estimating the kernel bandwidth.
 * --- bin : whether to bin the data or not. */
numerics::kde::kde(kernels k, bandwidth_estimator estim, bool bin) : bandwidth(bdw), data(x) {
    kern = k;
    method = estim;
    binning = bin;
    bdw = 0;
}

/* kde(h, k, bin) : initialize kde object by specifying a bandwidth.
 * --- h : bandwidth to use.
 * --- k : kernel to use.
 * --- bin : whether to bin the data or not. */
numerics::kde::kde(double h, kernels k, bool bin) : bandwidth(bdw), data(x) {
    if (h <= 0) {
        std::cerr << "kde::kde() error: invalid choice for bandwidth (must be > 0 but bdw recieved = " << h << ").\n";
        bdw = 0;
        method = bandwidth_estimator::min_sd_iqr;
    } else {
        bdw = h;
    }
    binning = bin;
}

/* kde(fname) : initialize kde object by loading in previously constructed and saved object. */
numerics::kde::kde(const std::string& fname) : bandwidth(bdw), data(x) {
    load(fname);
}

/* fit(data) : fit kernel density estimator.
 * data : data to fit kde over. */
numerics::kde& numerics::kde::fit(const arma::vec& data) {
    if (binning) bins.to_bins(data);
    x = arma::sort(data);
    stddev = arma::stddev(x);

    if (bdw <= 0) {
        if (method == bandwidth_estimator::rule_of_thumb_sd) {
            if (binning) bdw = bw::rot1(bins.n_bins, stddev);
            else bdw = bw::rot1(x.n_elem, stddev);
        } else if (method == bandwidth_estimator::min_sd_iqr) {
            if (binning) bdw = bw::rot2(x, bins.n_bins);
            else bdw = bw::rot2(x, stddev);
        } else if (method == bandwidth_estimator::direct_plug_in) {
            if (binning) bdw = bw::dpi_binned(bins, stddev, kern);
            else bdw = bw::dpi(x, stddev, kern);
        } else bdw = bw::grid_mse(x, kern, 0, 40, binning); // method == bandwidth::grid_cv
    }
    return *this;
}

/* sample(n=1) : produce random sample from density estimate.
 * --- n : sample size. */
arma::vec numerics::kde::sample(uint N) {
    arma::vec rvals;

    if (binning) rvals = bins.bins(sample_from(N, bins.counts/x.n_elem));
    else rvals = x( arma::randi<arma::uvec>(N, arma::distr_param(0,x.n_elem-1)) );

    arma::vec kern_rvals;
    if (kern == kernels::gaussian) {
        kern_rvals = arma::randn(N);
    } else if (kern == kernels::square) {
        kern_rvals = (arma::randu(N)-0.5);
    } else if (kern == kernels::triangle) {
        kern_rvals = arma::randu(N);
        arma::uvec less_half = arma::find(kern_rvals < 0.5);
        arma::uvec greater_half = arma::find(kern_rvals >= 0.5);
        kern_rvals(less_half) = -1 * arma::sqrt(2 * kern_rvals(less_half));
        kern_rvals(greater_half) = 1 - arma::sqrt(2 - 2*kern_rvals(greater_half));
    } else { // kern == kernels::parabolic
        kern_rvals = arma::randu(N);
        kern_rvals = 2*arma::sin(arma::asin(2*kern_rvals-1)/3); // inverse cdf for parabolic kernel
    }
    kern_rvals *= bdw;
    rvals += kern_rvals;
    return rvals;
}

/* predict(t) : predict density for new data.
 * --- t : query point. */
double numerics::kde::predict(double t) {
    arma::vec tt = {t};
    return predict(tt)(0);
}

/* operator()(t) : same as predict(t).
 * --- t : query point. */
double numerics::kde::operator()(double t) {
    return predict(t);
}

/* predict(t) : predict density for new data.
 * --- t : vector of query points. */
arma::vec numerics::kde::predict(const arma::vec& t) {
    if (bdw <= 0) {
        std::cerr << "kde::predict() error: the estimator was never fitted; returning empty vector.\n";
        return arma::vec();
    }
    arma::vec fhat = arma::zeros(arma::size(t));
    for (int i=0; i < fhat.n_elem; ++i) {
        arma::vec r;
        if (binning) r = arma::abs(bins.bins - t(i))/bdw;
        else r = arma::abs(x - t(i))/bdw;

        arma::vec K = bw::eval_kernel(r, kern);

        if (binning) fhat(i) = arma::dot(bins.counts, K) / (x.n_elem * bdw);
        else fhat(i) = arma::sum(K) / (x.n_elem * bdw);
    }
    return fhat;
}

/* operator()(t) : predict density for new data.
 * --- t : vector of query points. */
arma::vec numerics::kde::operator()(const arma::vec& t) {
    return predict(t);
}

/* save(fname="kde.kde") : save kde to file. */
void numerics::kde::save(const std::string& fname) {
    std::ofstream out(fname);
    out << x.n_elem << " " << bdw << " " << stddev << " " << (int)binning << " " << (int)kern << "\n";
    x.t().raw_print(out);
}

/* load(fname) : load kde object from file. */
void numerics::kde::load(const std::string& fname) {
    std::ifstream in(fname);
    int a,b,c;
    in >> a >> bdw >> stddev >> b >> c;
    if (a < 0 || b < 0 || c < 0 || c > 3) {
        std::cerr << "kde::load error: invalid entry in file resulted in premature termination of load().\n";
        return;
    }
    x = arma::vec(a);
    binning = (bool)b;
    kern = (kernels)c;
    for (int i=0; i < a; ++i) in >> x(i);
    if (binning) bins.to_bins(x);
}