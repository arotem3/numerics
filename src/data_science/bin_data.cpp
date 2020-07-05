#include <numerics.hpp>

/* to_bins(x) : place data into regularly spaced 'bins' with non-integer 'counts' corresponding to a linear distance weights.
 * i.e. if bin1 = 0 and bin2 = 1 and x = 0.2 then bin1 would have a count of 0.8 and bin2 would have a count of 0.2.
 * --- x : data to place into bins. */
void numerics::bin_data::to_bins(const arma::vec& x) {
    if (x.n_elem <= 2) {
        std::cerr << "BinData::to_bins() error: data vector must contain more than two elements.\n";
        return;
    }
    if (_n == 0) {
        if (x.n_elem > 1000) _n = 500; // almost always sufficient
        if (x.n_elem > 30) _n = x.n_elem / 10; // somewhat arbitrary... we want _n << x.n_elem
        else _n = x.n_elem / 5;
        if (_n <= 5) _n = x.n_elem;
    }
    double xmin = x.min();
    double xmax = x.max();
    double eps = 1e-3 * (xmax - xmin);
    _bins = arma::linspace(xmin - eps, xmax + eps, _n);
    double _bin_width = _bins(1) - _bins(0);
    _counts = arma::zeros(_n);

    arma::vec xx;
    if (x.is_sorted()) xx = x;
    else xx = arma::sort(x);
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

/* to_bins(x, y) : place data into regularly spaced 'bins' with non-integer 'counts' corresponding to a linear distance weights of the dependent variable y.
 * i.e. if bin1 = 0 and bin2 = 1 and x = 0.2 then bin1 would have a count of 0.8 and bin2 would have a count of 0.2.
 * --- x : data to place into bins.
 * --- y : dependant variable */
void numerics::bin_data::to_bins(const arma::vec& x, const arma::vec& y) {
    if (x.n_elem != y.n_elem) {
        std::cerr << "bin_data::to_bins() error: data vectors must have the same length (x.n_elem = " << x.n_elem << " != " << y.n_elem << " = y.n_elem).\n";
        return;
    }
    if (x.n_elem <= 2) {
        std::cerr << "bin_data::to_bins() error: data vector must contain more than two elements.\n";
        return;
    }
    if (_n == 0) {
        if (x.n_elem > 1000) _n = 500; // almost always sufficient
        else if (x.n_elem > 30) _n = x.n_elem / 10; // somewhat arbitrary... we want _n << x.n_elem
        else if (x.n_elem/5 > 5) _n = x.n_elem / 5;
        else _n = x.n_elem;
    }

    arma::vec xx, yy;
    if (x.is_sorted()) {
        xx = x;
        yy = y;
    } else {
        arma::uvec I = arma::sort_index(x);
        xx = x(I);
        yy = y(I);
    }

    double xmin = xx(0);
    double xmax = xx(x.n_elem-1);
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