#include "numerics.hpp"

numerics::CollocPoly::CollocPoly(const arma::vec& xx, const arma::mat& ff) : a(_a), b(_b) {
    _a = xx.min(); _b = xx.max();
    _x = 2*(xx - _a)/(_b - _a) - 1; // map to [-1, 1]
    _f = ff;

    _w = arma::ones(_x.n_elem);
    for (u_long i=0; i < _x.n_elem; ++i) {
        for (u_long j=0; j < _x.n_elem; ++j) {
            if (i != j) _w(i) *= _x(i) - _x(j);
        }
    }
    _w = 1/_w;
}

arma::mat numerics::CollocPoly::operator()(const arma::vec& xx) const {
    arma::vec z = 2*(xx - _a)/(_b - _a) - 1;
    
    arma::mat numer = arma::zeros(z.n_elem, _f.n_cols);
    arma::vec denom = arma::zeros(z.n_elem);

    std::map<u_long, u_long> exact;

    for (u_long j=0; j < _x.n_elem; ++j) {
        arma::vec xdiff = z - _x(j);

        arma::uvec exct = arma::find(xdiff == 0);
        for (arma::uword e : exct) exact[e] = j;

        arma::vec tmp = _w(j) / xdiff;
        numer += tmp * _f.row(j);
        denom += tmp;
    }

    arma::mat ff = std::move(numer);
    ff.each_col() /= denom;
    
    for (auto e : exact) ff.row(e.first) = _f.row(e.second);

    return ff;
}

numerics::ChebInterp::ChebInterp(u_long N, double aa, double bb, const std::function<arma::mat(double)>& func) {
    N--;
    _a = aa;
    _b = bb;
    _x = arma::cos(arma::regspace(0,N)*M_PI/N);
    
    double t = 0.5*(_x(0) + 1)*(_b - _a) + _a;
    arma::mat tmp = func(t);
    _f.set_size(N+1, tmp.n_elem);
    _f.row(0) = tmp.as_row();
    for (u_long i=1; i < N+1; ++i) {
        t = 0.5*(_x(i) + 1)*(_b - _a) + _a;
        _f.row(i) = func(t).as_row();
    }

    _w = arma::ones(N+1);
    _w(arma::regspace<arma::uvec>(1,2,N)) *= -1;
    _w(0) *= 0.5;
    _w(N) *= 0.5;
}

numerics::ChebInterp::ChebInterp(const arma::vec& xx, const arma::mat& ff) {
    _a = xx.min();
    _b = xx.max();
    _x = 2*(xx - _a)/(_b - _a) - 1;
    _f = ff;
    u_long N = xx.n_elem - 1;
    _w = arma::ones(N+1);
    _w(arma::regspace<arma::uvec>(1,2,N)) *= -1;
    _w(0) *= 0.5;
    _w(N) *= 0.5;
}