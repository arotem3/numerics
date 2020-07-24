#include <numerics.hpp>

/* integral_R(K) : compute the integral of K^2 over [-inf, inf]. R(K) is the notation in the literature.
 * --- K : Kernel name.
 * utility for dpi */
inline double integral_R(const std::string& K) {
    if (K == "gaussian") return 1/std::sqrt(4*M_PI);
    else if (K == "parabolic") return 0.6;
    else if (K == "square") return 0.5;
    else return 2.0/3; // triangular
}

/* second_moment(K) : compute the second moment of K.
 * --- K : kernel name.
 * utility for dpi */
inline double second_moment(const std::string& K) {
    if (K == "gaussian") return 1.0;
    else if (K == "parabolic") return 0.2;
    else if (K == "square") return 1.0/3;
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
double numerics::bw::dpi(const arma::vec& x, double s, const std::string& K) {
    check_kernel(K);
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
double numerics::bw::dpi_binned(const BinData& bins, double s, const std::string& K) {
    check_kernel(K);
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

arma::vec numerics::bw::eval_kernel(const arma::vec& r, const std::string& kern) {
    check_kernel(kern);
    arma::vec K;
    if (kern == "gaussian") {
        K = arma::exp(-0.5*arma::pow(r,2))/std::sqrt(2*M_PI);
    } else if (kern == "square") {
        K = arma::zeros(arma::size(r));
        K(arma::find(r <= 1)) += 0.5;
    } else if (kern == "triangle") {
        K = 1 - r;
        K(arma::find(K < 0)) *= 0;
    } else if (kern == "parabolic") {
        K = 0.75 * (1 - arma::pow(r,2));
        K(arma::find(K < 0)) *= 0;
    }
    return K;
}

/* smooth(x, y, t, h, kern) : apply a smoothing kernel to 't' using data 'x' and 'y'.
 * --- x : training data, independent variable
 * --- y : training data, dependent variable
 * --- t : points to evaluate smoothing function over.
 * --- h : kernel bandwidth.
 * --- kern : smoothing kernel.
 * utility function for class kernel_smooth */
arma::vec smooth(const arma::vec& x, const arma::vec& y, const arma::vec& t, double h, const std::string& kern) {
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
arma::vec smooth(const numerics::BinData& xy, const arma::vec& t, double h, const std::string& kern) {
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
double numerics::bw::grid_mse(const arma::vec& x, const arma::vec& y, const std::string& kern, double s, int grid_size, bool binning) {
    if (s <= 0) s = arma::stddev(x);
    double max_h = (x.max() - x.min())/4;
    double min_h = std::min(2*arma::median(arma::diff(arma::sort(x))), 0.05*s);
    
    arma::vec bdws = arma::logspace(std::log10(min_h), std::log10(max_h), grid_size);
    arma::vec cv_scores = arma::zeros(grid_size);

    int folds = 4;
    if (x.n_elem < 100) folds = 3;
    else if (x.n_elem < 30) folds = 2;
    numerics::KFolds2Arr<double,double> split(folds);
    split.fit(x,y);
    
    for (int j=0; j < folds; ++j) {
        BinData bins;
        if (binning) bins.fit(split.trainX(j), split.trainY(j));

        for (int i=0; i < grid_size; ++i) {
            arma::vec yhat;
            if (binning) yhat = smooth(bins, split.testX(j), bdws(i), kern);
            else yhat = smooth(split.trainX(j), split.trainY(j), split.testX(j), bdws(i), kern);
            cv_scores(i) += arma::norm(yhat - split.testY(j));
        }
    }

    u_int imin = cv_scores.index_min();
    return bdws(imin);
}

/* density(x, n, t, h, kern) : approximate density using kernel density estimation over binned data. (utility function for class kde)
 * --- x : bin_data object of prebinned data.
 * --- n : size of data.
 * --- t : points to evaluate density for.
 * --- h : bandwidth.
 * --- kern : kernel. */
arma::vec density(const numerics::BinData& x, int n, const arma::vec& t, double h, const std::string& kern) {
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
arma::vec density(const arma::vec& x, const arma::vec& t, double h, const std::string& kern) {
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
double numerics::bw::grid_mse(const arma::vec& x, const std::string& K, double s, int grid_size, bool binning) {
    if (s <= 0) s = arma::stddev(x);
    double max_h = (x(x.n_elem-1) - x(0))/4;
    double min_h = std::min(2*arma::median(arma::diff(x)), 0.05*s);

    arma::vec bdws = arma::logspace(std::log10(min_h), std::log10(max_h), grid_size);
    arma::vec cv_scores = arma::zeros(grid_size);

    int folds = 4;
    if (x.n_elem < 100) folds = 3;
    else if (x.n_elem < 30) folds = 2;
    numerics::KFolds1Arr<double> split(folds);
    split.fit(x);

    for (int j=0; j < folds; ++j) {
        BinData bins;
        if (binning) bins.fit(x);

        arma::vec pilot_density;
        if (binning) {
            pilot_density = density(bins, x.n_elem, split.test(j), bw::rot2(x), K);
            bins.fit(split.train(j));
        } else pilot_density = density(x, split.test(0), bw::rot2(x), K);

        for (int i=0; i < grid_size; ++i) {
            arma::vec pHat;
            if (binning) pHat = density(bins, x.n_elem, split.test(j), bdws(i), K);
            else pHat = density(split.train(0), split.test(j), bdws(i), K);
            cv_scores(i) += arma::norm(pHat - pilot_density);
        }
    }
    

    int imin = cv_scores.index_min();
    return bdws(imin);
}