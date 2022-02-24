#ifndef NUMERICS_DERIVATIVES_HPP
#define NUMERICS_DERIVATIVES_HPP

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

#include <cmath>
#include <complex>
#include <algorithm>
#include <string>
#include <concepts>
#include <stdexcept>

#include "numerics/concepts.hpp"

namespace numerics
{
    /* jacobian_diag(f, x, h, catch_zero, npt) : computes only the diagonal of a system of nonlinear equations.
    * --- f : f(x) system to approximate jacobian of.
    * --- x : vector to evaluate jacobian at.
    * --- h : finite difference step size. method is O(h^npt)
    * --- catch_zero : rounds near zero elements to zero.
    * --- npt : number of points to use in FD, npt = 1 uses f(x) and f(x+h), npt=2,4 do not use f(x) but instead use points f(x +/- h), f(x +/- 2h). Since npt=2 and npt=1 require the same number of f evals, npt 2 is used for its better accuracy. */
    template<class Vec, std::invocable<Vec> Func, typename Real = typename Vec::value_type>
    Vec jacobian_diag(Func f, const Vec& x, Real h = std::sqrt(std::numeric_limits<Real>::epsilon()), int npt=1)
    {
        u_long m = x.size();
        Vec J(m);
        Vec y = x;
        for (u_long i=0; i < m; ++i) {
            auto ff = [&f,&x,&y,i](Real z) -> Real
            {
                y[i] = z;
                Real f_y = f(y)[i];
                y[i] = x[i];
                return f_y;
            };
            J[i] = deriv(ff, x[i], h, npt);
        }
        return J;
    }

    template <class vec, std::invocable<vec> Func, typename real = typename vec::value_type>
    vec hessian_diag(Func f, const vec& x, real h = std::cbrt(std::numeric_limits<real>::epsilon()))
    {
        u_long m = x.size();
        vec B(m);
        vec y = x;
        real f0 = f(x);
        for (u_long i=0; i < m; ++i)
        {
            B[i] = -2*f0;
            real C = h * std::max<real>(std::abs(x[i]), 1);
            y[i] -= C;
            B[i] += f(y);
            y[i] = x[i] + C;
            B[i] += f(y);
            B[i] /= C*C; // (f(x-h) - 2*f(x) + f(x+h))/h^2
            y[i] = x[i];
        }

        return B;
    }

    /* grad(f, x, h, catch_zero, npt) : computes the gradient of a function of multiple variables.
    * --- f  : f(x) whose gradient to approximate.
    * --- x  : vector to evaluate gradient at.
    * --- h  : finite difference step size. method is O(h^npt).
    * --- catch_zero: rounds near zero elements to zero.
    * --- npt : number of points to use in FD, npt = 1 uses f(x) and f(x+h), npt=2,4 do not use f(x) but instead use points f(x +/- h), f(x +/- 2h). npt=1 is the default since it uses n+1 calls to f; npt=2 uses 2*n calls to f, and npt=4 uses 4*n calls to f. */
    template<class Vec, std::invocable<Vec> Func, scalar_field_type scalar = typename Vec::value_type>
    Vec grad(Func f, const Vec& x, precision_t<scalar> h=std::sqrt(std::numeric_limits<precision_t<scalar>>::epsilon()), int npt=1)
    {
        u_long n = x.size();
        Vec g(n);
        if (npt == 1) { // specialize this instance to minimize repeated calculations.
            scalar f0 = f(x);
            Vec y = x;
            for (u_long i=0; i < n; ++i)
            {
                scalar e = h*std::max<precision_t<scalar>>(std::abs(y[i]), 1.0f);
                y[i] += e;
                g[i] = (f(y) - f0) / e;
                y[i] = x[i];
            }
        } else {
            Vec y = x;
            for (u_long i=0; i < n; ++i)
            {
                auto ff = [&f,&x,&y,i](scalar t) -> scalar
                {
                    y[i] = t;
                    scalar f_y = f(y);
                    y[i] = x[i];
                    return f_y;
                };
                g[i] = deriv(ff, x[i], h, npt);
            }
        }
        
        return g;
    }

    /* deriv(f, x, h, catch_zero, npt) : computes the approximate derivative of a function of a single variable.
    * --- f  : f(x) whose derivative to approximate.
    * --- x  : point to evaluate derivative.
    * --- h  : finite difference step size; method is O(h^npt).
    * --- abstol: round anyvalue less than abstol to zero.
    * --- npt : number of points to use in FD, npt = 1 uses f(x) and f(x+h), npt=2,4 do not use f(x) but instead use points f(x +/- h), f(x +/- 2h). Since npt=2 and npt=1 require the same number of f evals, npt 2 is used for its better accuracy. */
    template<scalar_field_type T, std::invocable<T> Func>
    inline T deriv(Func f, const T& x, precision_t<T> h=std::sqrt(std::numeric_limits<precision_t<T>>::epsilon()), int npt=2) {
        T df;
        T e = h*std::max<precision_t<T>>(std::abs(x), 1.0f);
        if (npt == 1)
            df = (f(x + e) - f(x))/e;
        else if (npt == 2)
            df = (f(x + e) - f(x - e))/(T(2.0f)*e);
        else if (npt == 4)
            df = (f(x - T(2.0f)*e) - T(8.0f)*f(x - e) + T(8.0f)*f(x + e) - f(x + T(2.0f)*e))/(T(12.0f)*e);
        else
            throw std::invalid_argument("deriv() error: only 1, 2, and 4 point FD derivatives supported (but npt = " + std::to_string(npt) + " was provided).");
        
        return df;
    }

    namespace __directional_grad
    {
        #ifdef NUMERICS_WITH_ARMA
        template <typename Real>
        Real norm_impl(const arma::Mat<Real>& x)
        {
            return arma::norm(x,"inf");
        }
        #endif
        template <class vec, typename Real=typename vec::value_type>
        Real norm_impl(const vec& x)
        {
            Real norm_of_x = 0;
            for (u_long i=0; i < x.size(); ++i)
                norm_of_x += std::max<Real>(norm_of_x, std::abs(x[i]));
            return norm_of_x;
        }
    }

    /* directional_grad(f, x, h, catch_zero, npt) : approximates J*v for J = jacobian of f; i.e. the gradient of f in the direction of v. Uses vectorized arithmetic operations and a norm(x) function so cannot be applied to std::vector with out creating these operators and a norm function. This function is especially designed to work with armadillo types.
    * --- f  : f(x) whose derivative to approximate
    * --- x  : point to evaluate derivative at
    * --- v  : direction of derivative
    * --- abstol : rounds near zero elements to zero if less than abstol.
    * --- npt : number of points to use in FD. */
    template<class Vec, std::invocable<Vec> Func, scalar_field_type scalar = typename Vec::value_type>
    Vec directional_grad(Func f, const Vec& x, const Vec& v, precision_t<scalar> h=std::sqrt(std::numeric_limits<precision_t<scalar>>::epsilon()), int npt=2) {
        typedef precision_t<scalar> precision;
        
        if (x.size() != v.size()) {
            throw std::invalid_argument(
                "directional_grad() error: cannot compute derivative of f at x in the direction v when x.size() (="
                + std::to_string(x.size()) + ") does not match v.size() (="
                + std::to_string(v.size()) + ")."
            );
        }

        Vec Jv(x.size());

        const precision C = h * std::max<precision>(precision(1.0f), __directional_grad::norm_impl(x)) / std::max<precision>(1, __directional_grad::norm_impl(v));

        if (npt == 1)
            Jv = (f(x + C*v) - f(x)) / C;
        else if (npt == 2)
            Jv = (f(x + C*v) - f(x - C*v)) / (2*C);
        else if (npt == 4)
            Jv = (f(x - scalar(2*C)*v) - scalar(8)*f(x - C*v) + scalar(8)*f(x + C*v) - f(x + scalar(2*C)*v)) / (12*C);
        else
            throw std::invalid_argument("directional_grad() error: only 1, 2, and 4 point FD derivatives supported (but npt = " + std::to_string(npt) + " was provided).");

        return Jv;
    }

    #ifdef NUMERICS_WITH_ARMA
    /* approx_jacobian(f, x, h, catch_zero, npt) : computes the jacobian of a system of nonlinear equations.
    * --- f  : f(x) whose jacobian to approximate.
    * --- x  : vector to evaluate jacobian at.
    * --- h : finite difference step size. method is O(h^npt)
    * --- catch_zero: rounds near zero elements to zero.
    * --- npt : number of number of points to use in FD, npt = 1 uses f(x) and f(x+h), npt=2,4 do not use f(x) but instead use points f(x +/- h), f(x +/- 2h). npt=1 is the default since it uses n+1 calls to f; npt=2 uses 2*n calls to f, and npt=4 uses 4*n calls to f. */
    template<scalar_field_type scalar, std::invocable<arma::Col<scalar>> Func>
    arma::Mat<scalar> jacobian(Func f, const arma::Col<scalar>& x, precision_t<scalar> h=10*std::sqrt(std::numeric_limits<precision_t<scalar>>::epsilon()), int npt=1) {
        typedef precision_t<scalar> precision;

        u_long n = x.size(); // num variables -> num cols
        if (n < 1) throw std::invalid_argument("approx_jacobian() error: when computing the jacobian, require x.size() (=" + std::to_string(n) + ") >= 1.");

        arma::Mat<scalar> J;
        if (npt == 1) {
            arma::Col<scalar> fx = f(x);
            J.set_size(fx.n_elem, n);
            arma::Col<scalar> y = x;
            for (u_long i=0; i < n; ++i) {
                precision e = h * std::max<precision>(std::abs(y[i]), precision(1.0f));
                y[i] += e;
                J.col(i) = (f(y) - fx) / e; 
                y[i] = x[i];
            }
        }
        else if (npt == 2) {
            arma::Col<scalar> y = x;
            for (u_long i=0; i < n; ++i) {
                precision e = h * std::max<precision>(std::abs(y[i]), precision(1.0f));
                y[i] += e;
                arma::Col<scalar> df = f(y);
                y[i] = x[i] - e;
                df -= f(y);
                y[i] = x[i];
                
                df /= 2*e;
                if (J.empty())
                    J.set_size(df.size(), n);
                J.col(i) = df;
            }
        }
        else if (npt == 4) {
            arma::Col<scalar> y = x;
            for (u_long i=0; i < x.n_elem; ++i) {
                precision e = h * std::max<precision>(std::abs(y[i]), precision(1.0f));
                y[i] = x[i] + 2*e;
                arma::Col<scalar> df = -f(y);
                y[i] = x[i] + e;
                df += scalar(8.0f)*f(y);
                y[i] = x[i] - e;
                df -= scalar(8.0f)*f(y);
                y[i] = x[i] - 2*e;
                df += f(y);
                y[i] = x[i];

                df /= 12*e;
                if (J.empty())
                    J.set_size(df.size(), n);
                J.col(i) = df;
            }
        }
        else {
            throw std::invalid_argument("approx_jacobian() error: only 1, 2, and 4 point derivatives supported (not " + std::to_string(npt) + ").");
        }

        return J;
    }
    #endif // NUMERICS_WITH_ARMA
} // namespace nuemerics

#endif // NUMERICS_DERIVATIVES_HPP