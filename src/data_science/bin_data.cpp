#include <numerics.hpp>

void numerics::BinData::_set_bins(u_long n_obs) {
    if (_n == 0) {
        if (n_obs > 1000) _n = 500; // almost always sufficient
        if (n_obs > 30) _n = n_obs / 10; // somewhat arbitrary... we want _n << x.n_elem
        else _n = n_obs / 5;
        if (_n <= 5) _n = n_obs;
    }
    if (n_obs <= _n) {
        throw std::invalid_argument("require number of observations in x (x.n_rows=" + std::to_string(n_obs) + ") > number of bins (=" + std::to_string(_n) + ").");
    }
}

void numerics::BinData::fit(const arma::mat& x) {
    _set_bins(x.n_rows);
    
    if (x.n_cols > 1) {
        throw std::logic_error("multi-dimensional binning not yet supported (x.n_cols=" + std::to_string(x.n_cols) + " > 1).");
    } else {
        arma::vec xx = arma::sort(x);
        double xmin = xx.front();
        double xmax = xx.back();
        double eps = 1e-3 * (xmax - xmin);
        _bins = arma::linspace(xmin - eps, xmax + eps, _n);
        double _bin_width = _bins(1) - _bins(0);
        _counts = arma::zeros(_n);

        int j = 0;
        for (int i=0; i < xx.n_elem; ++i) {
            bool counted = false;
            while (!counted) {
                if (_bins(j) <= xx(i) && xx(i) < _bins(j+1)) { // xx(i) is in bin j
                    double w = (_bins(j+1) - xx(i)) / _bin_width;
                    _counts(j) += w;
                    _counts(j+1) += 1-w;
                    counted = true;
                } else j++;
            }
        }
    }
}

void numerics::BinData::fit(const arma::mat& x, const arma::vec& y) {
    if (x.n_rows != y.n_rows) {
        throw std::invalid_argument("data vectors must have the same length (x.n_rows = " + std::to_string(x.n_rows) + " != " + std::to_string(y.n_rows) + " = y.n_rows)");
    }
    _set_bins(x.n_rows);

    if (bins.n_cols > 1) {
        throw std::logic_error("multi-dimensional binning not yet supported (x.n_cols=" + std::to_string(x.n_cols) + " > 1).");
    } else {
        arma::uvec I = arma::sort_index(x);
        arma::vec xx = x(I);
        arma::vec yy = y(I);

        double xmin = xx.front();
        double xmax = xx.back();
        double eps = 1e-3 * (xmax - xmin);
        _bins = arma::linspace(xmin - eps, xmax + eps, _n);
        double _bin_width = _bins(1) - _bins(0);
        _counts = arma::zeros(_n);

        int i=0;
        double w=0, sum_w=0, sum_yw=0;
        while (i < x.n_elem && xx(i) < bins(1)) {
            w = 1 - std::abs(xx(i)-_bins(1))/_bin_width;
            sum_w += w;
            sum_yw += yy(i)*w;
            i++;
        }
        if (sum_w != 0) _counts(0) = sum_yw / sum_w;

        int i1;
        for (int j=1; j < _n-1; ++j) {
            sum_w=0;
            sum_yw=0;
            if (j==_n-2) i1 = i;
            while (i < x.n_elem && _bins(j-1) <= xx(i) && xx(i) < _bins(j+1)) {
                w = 1 - std::abs(xx(i)-_bins(j))/_bin_width;
                sum_w += w;
                sum_yw += yy(i)*w;
                i++;
            }
            if (sum_w != 0) _counts(j) = sum_yw/sum_w;
        }

        sum_w=0;
        sum_yw=0;
        i = i1;
        while (i < x.n_elem) {
            w = 1 - std::abs(xx(i)-_bins(_n-1))/_bin_width;
            sum_w += w;
            sum_yw += yy(i)*w;
            i++;
        }
        if (sum_w != 0) _counts(_n-1) = sum_yw/sum_w;
    }
}