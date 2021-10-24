#ifndef NUMERICS_INTERPOLATION_COLLOCPOLY_HPP
#define NUMERICS_INTERPOLATION_COLLOCPOLY_HPP

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

#include <cmath>
#include <unordered_map>
#include <string>

#ifdef NUMERICS_WITH_ARMA
namespace numerics
{

template <typename Real>
class CollocPoly
{
protected:
    arma::Col<Real> _x, _w;
    arma::Mat<Real> _f;
    Real _a, _b;

    CollocPoly() {}

public:
    typedef Real value_type;

    const Real& a = _a;
    const Real& b = _b;

    CollocPoly(CollocPoly<Real>&& p);
    CollocPoly(const CollocPoly& p);

    CollocPoly<Real>& operator=(CollocPoly<Real>&& p);
    CollocPoly<Real>& operator=(const CollocPoly<Real>& p);

    // implements stable evaluation of Lagrange polynomials using Barycentric
    // interpolation. Takes O(n^2) to construct and O(n) to evaluate. See
    // Berrut, Jean-Paul, and Lloyd N. Trefethen. “Barycentric Lagrange
    // Interpolation.” SIAM Review, vol. 46, no. 3, 2004, pp. 501–517.,
    // doi:10.1137/S0036144502417715.
    explicit CollocPoly(const arma::Col<Real>& xx, const arma::Mat<Real>& ff);

    arma::Mat<Real> operator()(const arma::Col<Real>& xx) const;
};

template <typename Real>
CollocPoly<Real>::CollocPoly(CollocPoly<Real>&& p)  {
    _x = std::move(p._x);
    _w = std::move(p._w);
    _f = std::move(p._f);
    _a = std::move(p._a);
    _b = std::move(p._b);
}

template <typename Real>
CollocPoly<Real>::CollocPoly(const CollocPoly<Real>& p)  {
    _x = p._x;
    _w = p._w;
    _f = p._f;
    _a = p._a;
    _b = p._b;
}

template <typename Real>
CollocPoly<Real>& CollocPoly<Real>::operator=(CollocPoly<Real>&& p) {
    _x = std::move(p._x);
    _w = std::move(p._w);
    _f = std::move(p._f);
    _a = std::move(p._a);
    _b = std::move(p._b);
    return *this;
}

template <typename Real>
CollocPoly<Real>& CollocPoly<Real>::operator=(const CollocPoly<Real>& p) {
    _x = p._x;
    _w = p._w;
    _f = p._f;
    _a = p._a;
    _b = p._b;
    return *this;
}

template <typename Real>
CollocPoly<Real>::CollocPoly(const arma::Col<Real>& xx, const arma::Mat<Real>& ff) {
    _a = xx.min(); _b = xx.max();
    _x = 2*(xx - _a)/(_b - _a) - 1; // map to [-1, 1]
    _f = ff;

    _w = arma::ones<arma::Col<Real>>(_x.n_elem);
    for (u_long i=0; i < _x.n_elem; ++i) {
        for (u_long j=0; j < _x.n_elem; ++j) {
            if (i != j) _w(i) *= _x(i) - _x(j);
        }
    }
    _w = 1/_w;
}

template <typename Real>
arma::Mat<Real> CollocPoly<Real>::operator()(const arma::Col<Real>& xx) const {
    arma::Col<Real> z = 2*(xx - _a)/(_b - _a) - 1;
    
    arma::Mat<Real> numer = arma::zeros<arma::Mat<Real>>(z.n_elem, _f.n_cols);
    arma::Col<Real> denom = arma::zeros<arma::Col<Real>>(z.n_elem);

    std::unordered_map<u_long, u_long> exact; // for evaluating exactly on the grid points

    for (u_long j=0; j < _x.n_elem; ++j) {
        arma::Col<Real> xdiff = z - _x(j);

        arma::uvec exct = arma::find(xdiff == 0);
        for (arma::uword e : exct) exact[e] = j;

        arma::Col<Real> tmp = _w(j) / xdiff;
        numer += tmp * _f.row(j);
        denom += tmp;
    }

    arma::Mat<Real> ff = std::move(numer);
    ff.each_col() /= denom;
    
    for (auto e : exact) ff.row(e.first) = _f.row(e.second);

    return ff;
}

template <typename Real>
class ChebInterp : public CollocPoly<Real>
{
public:
    // interpolate func on [a, b] using Chebyshev nodes. func should take a Real
    // and return arma::Mat<Real>
    template <class Func>
    explicit ChebInterp(u_long N, Real a, Real b, const Func& func);
    // construct interpolation from data, assuming xx are Chebyshev nodes. This
    // implementation differs from CollocPoly in that exact Barycentric weights
    // are used.
    explicit ChebInterp(const arma::Col<Real>& xx, const arma::Mat<Real>& ff);
};

template <typename Real>
template <class Func>
ChebInterp<Real>::ChebInterp(u_long N, Real aa, Real bb, const Func& func) : CollocPoly<Real>() {
    N--;
    this->_a = aa;
    this->_b = bb;
    this->_x = arma::cos(arma::regspace<arma::Col<Real>>(0,N)*M_PI/N);
    
    Real t = 0.5*(this->_x(0) + 1)*(this->_b - this->_a) + this->_a;
    arma::Mat<Real> tmp = func(t);
    this->_f.set_size(N+1, tmp.n_elem);
    this->_f.row(0) = tmp.as_row();
    for (u_long i=1; i < N+1; ++i) {
        t = 0.5*(this->_x(i) + 1)*(this->_b - this->_a) + this->_a;
        this->_f.row(i) = func(t).as_row();
    }

    this->_w = arma::ones<arma::Col<Real>>(N+1);
    this->_w(arma::regspace<arma::uvec>(1,2,N)) *= -1;
    this->_w(0) *= 0.5;
    this->_w(N) *= 0.5;
}

template <typename Real>
ChebInterp<Real>::ChebInterp(const arma::Col<Real>& xx, const arma::Mat<Real>& ff) : CollocPoly<Real>() {
    if (xx.empty())
        throw std::invalid_argument("ChebInterp error: xx is empty");
    if (xx.n_rows != ff.n_rows)
        throw std::invalid_argument(
            "ChebInterp error: xx.n_rows (=" + std::to_string(xx.n_rows)
            + ") != ff.n_rows (=" + std::to_string(ff.n_rows) + ")."
        );

    this->_a = xx.min();
    this->_b = xx.max();
    this->_x = 2*(xx - this->_a)/(this->_b - this->_a) - 1;
    this->_f = ff;
    u_long N = xx.n_elem - 1;
    this->_w = arma::ones<arma::Col<Real>>(N+1);
    this->_w(arma::regspace<arma::uvec>(1,2,N)) *= -1;
    this->_w(0) *= 0.5;
    this->_w(N) *= 0.5;
}

}
#endif
#endif