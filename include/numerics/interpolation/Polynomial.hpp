#ifndef NUMERICS_INTERPOLATION_POLYNOMIAL_HPP
#define NUMERICS_INTERPOLATION_POLYNOMIAL_HPP

#include <utility>
#include <iostream>

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

#ifdef NUMERICS_WITH_ARMA
namespace numerics
{

template <typename Real>
class Polynomial
{
protected:
    arma::Mat<Real> _p;
    u_long _deg;
    Real _a;
    Real _b;

    inline void _set_degree();

public:
    typedef Real value_type;
    const Real& a = _a;
    const Real& b = _b;
    const u_long& degree = _deg;
    const arma::Mat<Real>& coefficients = _p;

    // initialize polynomial to a constant
    inline explicit Polynomial(Real s = 0);

    // initialize polynomial by copying coefficient matrix (each column is a
    // separate polynomial). The polynomial is assumed to be defined with
    // respect to the variable: z=(x-c)/h, where c=0.5*(lb+ub) and h=ub-lb
    inline Polynomial(const arma::Mat<Real>& p, Real lb = -1.0, Real ub = 1.0);

    // initialize polynomial by copying coefficient matrix (each column is a
    // separate polynomial). The polynomial is assumed to be defined with
    // respect to the variable: z=2*(x-a)/(b-a)-1 i.e. transforms [lb,ub] to
    // [-1,1]
    inline Polynomial(arma::Mat<Real>&& p, Real lb = -1.0, Real ub = 1.0);

    // initialize polynomial by solving the least-squares problem:
    // P=argmin||y-P(x)|| such that P is a polynomial of degree deg
    Polynomial(const arma::Col<Real>& x, const arma::Mat<Real>& y, u_long deg);

    // initialize polynomial by interpolating the data.
    Polynomial(const arma::Col<Real>& x, const arma::Mat<Real>& y);

    Polynomial(Polynomial<Real>&& P);
    Polynomial<Real>& operator=(Polynomial<Real>&& P);

    Polynomial(const Polynomial& P);
    Polynomial& operator=(const Polynomial& P);

    template<typename T>
    operator Polynomial<T>() const {
        return Polynomial<T>(arma::conv_to<arma::Mat<T>>::from(coefficients), static_cast<T>(a), static_cast<T>(b));
    }

    arma::Mat<Real> operator()(const arma::Col<Real>& x) const;

    #ifdef NUMERICS_INTERPOLATION_POLYDER_HPP
    Polynomial<Real> derivative(u_long k=1) const {
        if (_p.n_rows <= k) {
            return Polynomial<Real>(arma::zeros(1, _p.n_cols), a, b);
        }
        else {
            arma::mat dp(_p.n_rows-k, _p.n_cols);
            for (u_long i=0; i < _p.n_cols; ++i)
                dp.col(i) = polyder<arma::Col<Real>>(_p.col(i), k) * 2/(b - a);
            return Polynomial<Real>(std::move(dp), a, b);
        }
    }
    #endif

    #ifdef NUMERICS_INTERPOLATION_POLYINT_HPP
    Polynomial<Real> integral() const {
        arma::mat ip(_p.n_rows+1, _p.n_cols);
        for (u_long i=0; i < _p.n_cols; ++i)
            ip.col(i) = polyint<arma::Col<Real>>(_p.col(i)) * (b - a) / 2;
        return Polynomial<Real>(std::move(ip), a, b);
    }
    #endif
};

template <typename Real>
inline void Polynomial<Real>::_set_degree() {
    _deg = _p.n_rows - 1;
}

template <typename Real>
inline Polynomial<Real>::Polynomial(Real s) {
    _p = {s};
    _a = -1;
    _b = 1;
    _set_degree();
}

template <typename Real>
inline Polynomial<Real>::Polynomial(const arma::Mat<Real>& p, Real lb, Real ub) {
    _p = p;
    _a = aa;
    _b = bb;
    _set_degree();   
}

template <typename Real>
inline Polynomial<Real>::Polynomial(arma::Mat<Real>&& p, Real lb, Real ub) {
    _p = std::move(p);
    _a = aa;
    _b = bb;
    _set_degree();
}

template <typename Real>
Polynomial<Real>::Polynomial(const arma::Col<Real>& x, const arma::Mat<Real>& y, u_long deg) {
    _p.set_size(deg+1, y.n_cols);
    _a = x.min();
    _b = x.max();
    arma::vec xx = 2*(x - _a)/(_b - _a) - 1;
    for (u_long i=0; i < y.n_cols; ++i) _p.col(i) = arma::polyfit(xx, y.col(i), deg);
    _set_degree();
}

template <typename Real>
Polynomial<Real>::Polynomial(const arma::Col<Real>& x, const arma::Mat<Real>& y) {
    u_long n = x.n_elem;
    _p.set_size(n, y.n_cols);
    _a = x.min();
    _b = x.max();
    arma::vec xx = 2*(x - _a)/(_b - _a) - 1;
    for (u_long i=0; i < y.n_cols; ++i) _p.col(i) = arma::polyfit(xx, y.col(i), n-1);
    _set_degree();
}

template <typename Real>
Polynomial<Real>::Polynomial(Polynomial<Real>&& P) {
    _p = std::move(P._p);
    _deg = std::move(P._deg);
    _a = std::move(P._a);
    _b = std::move(P._b);
}

template <typename Real>
Polynomial<Real>& Polynomial<Real>::operator=(Polynomial&& P) {
    _p = std::move(P._p);
    _deg = std::move(P._deg);
    _a = std::move(P._a);
    _b = std::move(P._b);

    return *this;
}

template <typename Real>
Polynomial<Real>::Polynomial(const Polynomial& P) {
    _p = P._p;
    _a = P._a;
    _b = P._b;
    _set_degree();
}

template <typename Real>
Polynomial<Real>& Polynomial<Real>::operator=(const Polynomial& P) {
    _p = P._p;
    _a = P._a;
    _b = P._b;
    _set_degree();

    return *this;
}

template <typename Real>
arma::Mat<Real> Polynomial<Real>::operator()(const arma::Col<Real>& x) const {
    arma::Mat<Real> out(x.n_elem, _p.n_cols);
    arma::Col<Real> xx = 2*(x - _a)/(_b - _a) - 1;
    for (u_long i=0; i < _p.n_cols; ++i)
        out.col(i) = arma::polyval(_p.col(i), xx);
    return out;
}

// transform polynomial from z=2*(x-a)/(b-a)-1 to w=2*(x-aa)/(bb-aa)-1
template <typename Real>
Polynomial<Real> transform_poly(const Polynomial<Real>& P, Real aa, Real bb) {
    Real c = (bb - aa) / (P.b - P.a);
    Real d = aa - c*P.a;
    
    arma::Mat<Real> T = arma::zeros(degree + 1, degree + 1);
    for (u_long i=0; i < degree+1; ++i) {
        for (u_long j=i; j < degree+1; ++j) {
            Real jchoosei = (i < j-i) ? (i * std::beta(i, j-i+1)) : ((j-i)*std::beta(i+1, j-i));
            jchoosei = 1 / jchoosei;
            T(i, j) = jchoosei * std::pow(c, i) * std::pow(d, j-i);
        }
    }

    arma::Mat<Real> pp = arma::reverse(T * arma::reverse(P.coefficients));
    return Polynomial<Real>(std::move(pp), aa, bb);
}

template <typename T1, typename T2, typename T3=decltype(std::declval<T1>() + std::declval<T2>())>
Polynomial<T3> operator+(const Polynomial<T1>& P, const Polynomial<T2>& Q) {
    if (P.coefficients.n_cols != Q.coefficients.n_cols) {
        throw std::runtime_error("Polynomial addition error: vector polynomials must have the same number of dimensions");
    }

    arma::Mat<T3> Qc;
    if ((P.a == Q.a) and (P.b == Q.b))
        Qc = arma::conv_to<arma::Mat<T3>::from(Q.coefficients);
    else {
        Polynomial<T2> QQ = transform_poly(Q, P.a, P.b);
        Qc = arma::conv_to<arma::Mat<T3>::from(QQ.coefficients);
    }

    arma::Mat<T3> Pc = arma::conv_to<arma::Mat<T3>>::from(P.coefficients);
    
    u_long k = std::max(P.coefficients.n_rows, Qc.n_rows);
    
    arma::Mat<T3> psum = arma::zeros<arma::Mat<T3>>(k, P.n_cols);
    psum.tail_rows(P.degree+1) += Pc;
    psum.tail_rows(QQ.degree+1) += Qc;

    return Polynomial<T3>(std::move(psum), static_cast<T3>(P.a), static_cast<T3>(P.b));
}

template <typename T1, typename T2, typename T3=decltype(std::declval<T1>() + std::declval<T2>())>
Polynomial<T3> operator+(const Polynomial<T1>& P, const T2& c) {
    arma::Mat<T3> psum = arma::conv_to<arma::Mat<T3>>::from(P.coefficients);
    psum.tail_rows(1) += c;

    return Polynomial<T3>(std::move(psum), static_cast<T3>(P.a), static_cast<T3>(P.b));
}

template <typename T1, typename T2, typename T3=decltype(std::declval<T1>() + std::declval<T2>())>
Polynomial<T3> operator+(const T1& c, const Polynomial<T2>& P) {
    return P+c;
}

template <typename Real>
Polynomial<Real> operator-(const Polynomial<Real>& P) {
    return Polynomial<Real>(-P.coefficients, P.a, P.b);
}

template <typename T1, typename T2, typename T3=decltype(std::declval<T1>() - std::declval<T2>())>
Polynomial<T3> operator-(const Polynomial<T1>& P, const Polynomial<T2>& Q) {
    return P + (-Q);
}

template <typename T1, typename T2, typename T3=decltype(std::declval<T1>() - std::declval<T2>())>
Polynomial<T3> operator-(const Polynomial<T1>& P, const T2& c) {
    return P + (-c);
}

template <typename T1, typename T2, typename T3=decltype(std::declval<T1>() - std::declval<T2>())>
Polynomial<T3> operator-(const T1& c, const Polynomial<T2>& P) {
    return (-P) + c;
}

template <typename T1, typename T2, typename T3=decltype(std::declval<T1>() * std::declval<T2>())>
Polynomial<T3> operator*(const Polynomial<T1>& P, const Polynomial<T2>& Q) {
    if (P.coefficients.n_cols != Q.coefficients.n_cols) {
        throw std::runtime_error("Polynomial multiplication error: vector polynomials must have the same number of dimensions");
    }

    Polynomial<T2> QQ = transform_poly(Q, P.a, P.b);

    arma::Mat<T3> Pc = arma::conv_to<arma::Mat<T3>>::from(P.coefficients);
    arma::Mat<T3> Qc = arma::conv_to<arma::Mat<T3>>::From(Q.coefficients);

    arma::Mat<T3> pprod(P.degree + Q.degree + 2, P.coefficients.n_cols);
    for (u_long i=0; i < P.coefficients.n_cols; ++i)
        pprod.col(i) = arma::conv(Pc.col(i), Qc.col(i), "full");
    
    return Polynomial<T3>(std::move(pprod), static_cast<T3>(P.a), static_cast<T3>(P.b));
}

template <typename T1, typename T2, typename T3=decltype(std::declval<T1>() * std::declval<T2>())>
Polynomial<T3> operator*(const Polynomial<T1>& P, const T2& c) {
    return Polynomial<T3>(arma::conv_to<arma::Mat<T3>>::from(P.coefficients)*static_cast<T3>(c), static_cast<T3>(P.a), static_cast<T3>(P.b));
}

template <typename T1, typename T2, typename T3=decltype(std::declval<T1>() * std::declval<T2>())>
Polynomial<T3> operator*(const T1& c, const Polynomial<T2>& P) {
    return P * c;
}

template <typename Real>
std::ostream& operator<<(std::ostream& out, const Polynomial<Real>& p) {
    for (int i=0; i < p.degree+1; ++i) {
        out << p.coefficients.row(i);
        if (i < p.degree) out << " * (" << 2/(p.b-p.a) << " * x + " << 1 - p.a / (p.b - p.a) << ")";
        if (i < p.degree-1) out << "^" << p.degree-i;
        if (i < p.degree) out << " + ";
    }
    return out;
}

}

#endif
#endif