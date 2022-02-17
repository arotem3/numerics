#ifndef NUMERICS_INTERPOLATION_COLLOCPOLY_HPP
#define NUMERICS_INTERPOLATION_COLLOCPOLY_HPP

#include <cmath>
#include <string>
#include <vector>

namespace numerics {
    template <std::floating_point real, typename vec>
    class CollocPoly
    {
    protected:
        real _a, _b;
        std::vector<real> _x, _w;
        std::vector<vec> _f;

        CollocPoly() {}

    public:
        const real& a = _a;
        const real& b = _b;

        typedef real value_type;

        CollocPoly(CollocPoly<real,vec>&&);
        CollocPoly(const CollocPoly<real,vec>&);

        CollocPoly<real,vec>& operator=(CollocPoly<real,vec>&&);
        CollocPoly<real,vec>& operator=(const CollocPoly<real,vec>&);

        // implements stable evaluation of Lagrange polynomials using Barycentric
        // interpolation. Takes O(n^2) to construct and O(n) to evaluate. See
        // Berrut, Jean-Paul, and Lloyd N. Trefethen. “Barycentric Lagrange
        // Interpolation.” SIAM Review, vol. 46, no. 3, 2004, pp. 501–517.,
        // doi:10.1137/S0036144502417715.
        template <typename real_it, typename vec_it>
        explicit CollocPoly(real_it x_first, real_it x_last, vec_it f_first, vec_it f_last);

        // evaluate the interpolation
        vec operator()(real) const;
    };

    template <std::floating_point real, typename vec>
    CollocPoly<real,vec>::CollocPoly(CollocPoly<real,vec>&& p)  {
        _x = std::move(p._x);
        _w = std::move(p._w);
        _f = std::move(p._f);
        _a = p._a;
        _b = p._b;
    }

    template <std::floating_point real, typename vec>
    CollocPoly<real,vec>::CollocPoly(const CollocPoly<real,vec>& p)  {
        _x = p._x;
        _w = p._w;
        _f = p._f;
        _a = p._a;
        _b = p._b;
    }

    template <std::floating_point real, typename vec>
    CollocPoly<real,vec>& CollocPoly<real,vec>::operator=(CollocPoly<real,vec>&& p) {
        _x = std::move(p._x);
        _w = std::move(p._w);
        _f = std::move(p._f);
        _a = p._a;
        _b = p._b;
        return *this;
    }

    template <std::floating_point real, typename vec>
    CollocPoly<real,vec>& CollocPoly<real,vec>::operator=(const CollocPoly<real,vec>& p) {
        _x = p._x;
        _w = p._w;
        _f = p._f;
        _a = p._a;
        _b = p._b;
        return *this;
    }

    template <std::floating_point real, typename vec>
    template <typename real_it, typename vec_it>
    CollocPoly<real,vec>::CollocPoly(real_it x_first, real_it x_last, vec_it f_first, vec_it f_last)
    {
        for (; x_first != x_last; ++x_first)
            _x.push_back(*x_first);
        for (; f_first != f_last; ++f_first)
            _f.push_back(*f_first);

        auto [minit, maxit] = std::minmax_element(_x.begin(), _x.end());
        _a = *minit;
        _b = *maxit;

        _w.assign(_x.size(), real(1));

        real w_max = 0;
        real w_min = 0;

        auto wi = _w.begin();
        for (auto xi = _x.cbegin(); xi != _x.cend(); ++xi, ++wi)
        {
            for (auto xj = _x.cbegin(); xj != _x.cend(); ++xj)
            {
                if (xi != xj)
                    (*wi) *= (*xi) - (*xj);
            }
            
            (*wi) = 1 / (*wi);
            w_max = std::max<real>(w_max, *wi);
            w_min = std::min<real>(w_min, *wi);
        }

        real D = 1 / (w_max - w_min);

        std::for_each(_w.begin(), _w.end(), [D](real& z)->void{z *= D;});
    }

    template <std::floating_point real, typename vec>
    vec CollocPoly<real,vec>::operator()(real x) const
    {
        vec numer = real(0) * _f.front();
        real denom = 0;

        auto xi = _x.begin();
        auto fi = _f.begin();
        auto wi = _w.begin();

        for (; xi != _x.end(); ++xi, ++fi, ++wi)
        {
            real xdiff = x - (*xi);
            if (xdiff == 0)
                return *fi;
            
            real C = (*wi) / xdiff;
            numer += C * (*fi);
            denom += C;
        }

        numer = (1/denom) * numer;

        return numer;
    }

    template <std::floating_point real, typename vec>
    class ChebInterp : public CollocPoly<real, vec>
    {
    public:
        // interpolate func on [a, b] using Chebyshev nodes.
        template <std::invocable<real> Func>
        explicit ChebInterp(u_long N, real a, real b, Func func);

        // construct interpolation from data, assuming xx are Chebyshev nodes. This
        // implementation differs from CollocPoly in that exact Barycentric weights
        // are used.
        template <typename real_it, typename vec_it>
        explicit ChebInterp(real_it x_first, real_it x_last, vec_it f_first, vec_it f_last);
    };

    template <std::floating_point real, typename vec>
    template <std::invocable<real> Func>
    ChebInterp<real,vec>::ChebInterp(u_long N, real a, real b, Func func) : CollocPoly<real,vec>() {
        if (N < 2)
            throw std::invalid_argument("ChebInterp error: cannot construct interpolation on fewer than two points.");
        
        N--;
        this->_x.resize(N+1);
        this->_f.resize(N+1);
        this->_w.resize(N+1);
        this->_a = a;
        this->_b = b;

        auto xi = this->_x.begin();
        auto fi = this->_f.begin();
        auto wi = this->_w.begin();
        for (u_long i=0; i <= N; ++i, ++xi, ++fi, ++wi)
        {
            real t = -std::cos(i * M_PI / N);
            t = 0.5*(t + 1) * (b - a) + a;
            *xi = t;
            *fi = func(t);
            *wi = (i & 1) ? -1 : 1;
        }

        this->_w.front() *= 0.5;
        this->_w.back() *= 0.5;
    }

    template <std::floating_point real, typename vec>
    template <typename real_it, typename vec_it>
    ChebInterp<real,vec>::ChebInterp(real_it x_first, real_it x_last, vec_it f_first, vec_it f_last) : CollocPoly<real,vec>() {
        for (; x_first != x_last; ++x_first)
            this->_x.push_back(*x_first);
        for (; f_last != f_last; ++f_first)
            this->_f.push_back(*f_first);

        if (this->_x.size() < 2)
            throw std::invalid_argument("ChebInterp error: cannot construct interpolation on fewer than two points.");

        std::tie(this->_a, this->_b) = std::minmax({this->_x.front(), this->_x.back()});

        u_long N = this->_x.size() - 1;
        this->_w.resize(N+1);

        auto wi = this->_w.begin();
        for (size_t i=0; i <= N; ++i, ++wi)
            *wi = (i & 1) ? -1 : 1;
        this->_w.front() *= 0.5;
        this->_w.back() *= 0.5;
    }
} //namespace numerics
#endif