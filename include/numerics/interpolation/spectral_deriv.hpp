#ifndef NUMERICS_INTERPOLATION_SPECTRAL_DERIV_HPP
#define NUMERICS_INTERPOLATION_SPECTRAL_DERIV_HPP

#include <vector>
#include <stdexcept>
#include <concepts>
#include "numerics/interpolation/CollocPoly.hpp"

namespace numerics
{
    namespace __fftw
    {
        #include <fftw3.h>
    } // namespace __fftw

    // fills dy with the estimated derivative of y using the fast chebyshev transform
    // (via the fast discrete cosine transform) where y and dy are assumed to be
    // allocated arrays of size N. The points y are assumed to be evaluated at the
    // Chebyshev nodes: cos( k * pi / (N-1) ) for k=0,1,...,N-1 (which is on the
    // interval [-1, 1]). If the interval is mapped (affine) to [a,b], then dy will
    // need to be multiplied by 2/(b-a). Note that this array is reversed, i.e. it
    // is decreasing from 1 to -1, this is intentionally designed for compatibility
    // with the fast discrete cosine transform. The arrays yh and dyh (allocated to
    // size N) are the output arrays for fftw_plan instances dct and idct,
    // specifically: dct and idct are both REDFT00 r2r 1d transforms, but dct takes
    // y as input and yh as output (preserving the input), and idct takes dyh as
    // input and dy as output (not necessarily preserving the ouput). The arrays yh
    // and dyh as well as the plans are specified here so that repeated computations
    // of the derivative on the same array y is computed as efficiently as possible.
    // When calling this function for the first time, initialize dct and idct as
    // nullptr, and they will be initialized by the function. It is the
    // responsibility of the user to manage all of the pointers including dct and
    // idct which may be allocated within the function (call fftw_destroy_plan to
    // free those pointers).
    // The implementation is based on algorithm 40 and related discussion in chapter
    // 3 of "Implementing Spectral Methods for Partial Differential Equations" by D.
    // Kopriva
    // Frigo, M., and S. G. Johnson. “FFTW: An Adaptive Software Architecture
    // for the FFT.”.
    void cheb_deriv(u_long N, double * dy, double * y, double * yh, double * dyh, __fftw::fftw_plan& dct, __fftw::fftw_plan& idct)
    {
        if (N < 3)
            throw std::invalid_argument("cheb_deriv() error: require N >= 3; why? The Discrete Cosine Transform is not defined for N < 3.");

        // discrete cosine transform
        if (dct == nullptr)
            dct = __fftw::fftw_plan_r2r_1d(N, y, yh, __fftw::FFTW_REDFT00, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
        
        __fftw::fftw_execute(dct);

        // DCT normalization
        for (double * yi = yh; yi != yh+N; ++yi)
            *yi /= double(N-1);
        
        // DCT to Cheb coefs
        yh[0] *= 0.5;
        yh[N-1] *= 0.5;

        // derivative recurrence
        dyh[N-1] = 0;
        dyh[N-2] = 2*(N-1)*yh[N-1];
        for (u_long k = N-3; k > 0; --k)
            dyh[k] = double(2*(k+1)) * yh[k+1] + dyh[k+2];
        dyh[0] = yh[1] + 0.5 * dyh[2];

        // Cheb to DCT coef
        dyh[0] *= 2.0;
        dyh[N-1] *= 2.0;

        // (inverse) discrete cosine transform
        if (idct == nullptr)
            idct = __fftw::fftw_plan_r2r_1d(N, dyh, dy, __fftw::FFTW_REDFT00, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);

        __fftw::fftw_execute(idct);

        // DCT normalization and then we are done
        for (double * yi = dy; yi != dy + N; ++yi)
            (*yi) *= 0.5;
    }

    // fills dy with the estimated derivative of y using the fast chebyshev transform
    // (via the fast discrete cosine transform) where y and dy are assumed to be
    // allocated arrays of size N. The points y are assumed to be evaluated at the
    // Chebyshev nodes: cos( k * pi / (N-1) ) for k=0,1,...,N-1 (which is on the
    // interval [-1, 1]). If the interval is mapped (affine) to [a,b], then dy will
    // need to be multiplied by 2/(b-a). Note that this array is reversed, i.e. it
    // is decreasing from 1 to -1, this is intentionally designed for compatibility
    // with the fast discrete cosine transform.
    // The implementation is based on algorithm 40 and related discussion in chapter
    // 3 of "Implementing Spectral Methods for Partial Differential Equations" by D.
    // Kopriva
    // Frigo, M., and S. G. Johnson. “FFTW: An Adaptive Software Architecture
    // for the FFT.”.
    void cheb_deriv(u_long N, double * dy, double * y)
    {
        __fftw::fftw_plan dct = nullptr;
        __fftw::fftw_plan idct = nullptr;

        std::vector<double> yh(N);
        std::vector<double> dyh(N);

        cheb_deriv(N, dy, y, yh.data(), dyh.data(), dct, idct);

        __fftw::fftw_destroy_plan(dct);
        __fftw::fftw_destroy_plan(idct);
    }

    // constructs a polynomial approximation of the derivative of the
    // vector-valued function f on the interval [a,b] using the fast chebyshev
    // transform. The polynomial approximation will be of degree sample_points-2
    // because the function is first approximated on grid of size sample_points,
    // which produced a polynomial of degree sample_points-1, and the derivative
    // is necessarily one order less. The vector type `vec` must behave like a
    // mathematical vector, that is, it is equipped with an addition operator
    // {+,+=} and an (right) scalar multiplication {*}. Unlike most other
    // functions in numerics, it is also required that vec behave like an
    // iterable standard container, that is, it must have a .size(), .begin(),
    // and .end() member functions, and dereferencing its iterator must be
    // convertible to a double because the implementation of this function uses
    // FFTW3, so the data is copied to C-style arrays.
    // The implementation is based on algorithm 40 and related discussion in chapter
    // 3 of "Implementing Spectral Methods for Partial Differential Equations" by D.
    // Kopriva
    // Frigo, M., and S. G. Johnson. “FFTW: An Adaptive Software Architecture
    // for the FFT.”.
    template <typename vec, std::invocable<double> Func>
    ChebInterp<double, vec> spectral_deriv(Func f, double a, double b, u_long sample_points=32ul)
    {
        vec y = f(b);
        u_long dim = y.size();

        std::vector<double> x(sample_points);
        x[0] = b;
        std::vector<std::vector<double>> Y(dim);
        for (auto Yi = Y.begin(); Yi != Y.end(); ++Yi)
            Yi->resize(sample_points);

        // keep iterators for each vector in Y for better caching
        std::vector<std::vector<double>::iterator> ys(dim);
        auto yi = std::begin(y);
        for (u_long i=0; i < dim; ++i, ++yi)
        {
            ys[i] = Y[i].begin();
            *(ys[i]) = *yi;
            ++(ys[i]);
        }

        // evaluate f on cheb grid
        double h = 0.5 * (b - a);
        double c = 0.5 * a + 0.5 * b;
        auto xi = x.begin(); ++xi;
        for (u_long k=1; k < sample_points; ++k, ++xi)
        {
            double z = double(k) * M_PI / double(sample_points-1);
            z = h*std::cos(z) + c;
            vec y = f(z);
            (*xi) = z;

            auto yi = std::begin(y);
            for (auto ysi = std::begin(ys); ysi != std::end(ys); ++ysi, ++yi)
            {
                *(*ysi) = (*yi);
                ++(*ysi);
            }
        }

        // compute derivative
        std::vector<std::vector<double>> dY(dim);
        auto Yi = std::begin(Y);
        for (auto dYi = std::begin(dY); dYi != std::end(dY); ++dYi, ++Yi)
        {
            dYi->resize(sample_points);
            cheb_deriv(sample_points, dYi->data(), Yi->data());

            // rescale for interval
            for (auto p = dYi->begin(); p != dYi->end(); ++p)
                (*p) /= h;
        }

        // copy to ChebInterp object
        struct __it
        {
            std::vector<std::vector<double>::iterator> its;

            __it() {}

            __it& operator++()
            {
                for (auto ii = its.begin(); ii != its.end(); ++ii)
                    ++(*ii);

                return *this;
            }

            vec operator*() const
            {
                vec x(its.size());

                auto ii = its.begin();
                for (auto xi = std::begin(x); xi != std::end(x); ++xi, ++ii)
                    (*xi) = *(*ii);

                return x;
            }

            bool operator!=(const __it& it)
            {
                return ( its.front() != it.its.front() );
            }
        };

        __it dy_begin;
        dy_begin.its.resize(dim);
        auto jj = dY.begin();
        for (auto ii = dy_begin.its.begin(); ii != dy_begin.its.end(); ++ii, ++jj)
            (*ii) = jj->begin();
        
        __it dy_end;
        dy_end.its.push_back(Y[0].end()); // cheat

        return ChebInterp<double, vec>(x.begin(), x.end(), dy_begin, dy_end);
    }

    // constructs a polynomial approximation of the derivative of the
    // real-valued function f on the interval [a,b] using the fast chebyshev
    // transform. The polynomial approximation will be of degree sample_points-2
    // because the function is first approximated on grid of size sample_points,
    // which produced a polynomial of degree sample_points-1, and the derivative
    // is necessarily one order less. The implementation uses FFTW3.
    // The implementation is based on algorithm 40 and related discussion in chapter
    // 3 of "Implementing Spectral Methods for Partial Differential Equations" by D.
    // Kopriva
    // Frigo, M., and S. G. Johnson. “FFTW: An Adaptive Software Architecture
    // for the FFT.”.
    template <std::invocable<double> Func>
    ChebInterp<double,double> spectral_deriv(Func f, double a, double b, u_long sample_points=32ul)
    {
        std::vector<double> x(sample_points);
        std::vector<double> y(sample_points);
        std::vector<double> dy(sample_points);

        double h = 0.5 * (b - a);
        double c = 0.5 * a + 0.5 * b;
        auto xi = x.begin();
        auto yi = y.begin();
        for (u_long k=0; k < sample_points; ++k, ++xi, ++yi)
        {
            double z = double(k) * M_PI / double(sample_points-1);
            z = h*std::cos(z) + c;
            (*xi) = z;
            (*yi) = f(z);
        }

        cheb_deriv(sample_points, dy.data(), y.data());

        for (auto dyi = dy.begin(); dyi != dy.end(); ++dyi)
            (*dyi) /= h;

        return ChebInterp<double,double>(x.begin(), x.end(), dy.begin(), dy.end());
    }
} // namespace numerics

#endif