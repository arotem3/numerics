#ifndef NUMERICS_VMATH_HPP
#define NUMERICS_VMATH_HPP

#include "numerics/concepts.hpp"

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

namespace numerics
{
    namespace __vmath
    {
        #ifdef NUMERICS_WITH_ARMA
        template <scalar_field_type eT>
        inline precision_t<eT> dot_impl(const arma::Mat<eT>& x, const arma::Mat<eT>& y)
        {
            return arma::cdot(x, y);
        }

        template <scalar_field_type eT>
        inline precision_t<eT> norm_impl(const arma::Mat<eT>& x)
        {
            return arma::norm(x);
        }
        #endif

        // template <typename real, typename = typename std::enable_if<std::is_arithmetic<real>::value, real>::type>
        template <std::floating_point T>
        inline T dot_impl(T x, T y)
        {
            return x*y;
        }

        template <std::floating_point T>
        inline std::complex<T> dot_impl(std::complex<T> x, std::complex<T> y)
        {
            return std::conj(x) * y;
        }

        template <class vec, scalar_field_type T = typename vec::value_type>
        T dot_impl(const vec& x, const vec& y)
        {
            T dot = 0;
            for (u_long i=0; i < x.size(); ++i)
                dot += dot_impl(x[i], y[i]);
            return dot;
        }

        template <std::floating_point T>
        inline T norm_impl(T x)
        {
            return std::abs(x);
        }

        template <std::floating_point T>
        inline T norm_impl(std::complex<T> x)
        {
            return std::abs(x);
        }

        template <class vec, std::floating_point precision = precision_t<typename vec::value_type>>
        inline precision norm_impl(const vec& x)
        {
            return std::sqrt(std::abs(dot_impl(x,x)));
        }

        #ifdef NUMERICS_WITH_ARMA
        template <scalar_field_type eT>
        inline bool solve_impl(arma::Col<eT>& x, const arma::Mat<eT>& A, const arma::Col<eT>& b)
        {
            return arma::solve(x, A, b);
        }

        template <scalar_field_type eT>
        inline bool solve_impl(arma::Col<eT>& x, const arma::SpMat<eT>& A, const arma::Col<eT>& b)
        {
            return arma::spsolve(x, A, b);
        }
        #endif
    } // namespace __vmath
} // namespace numerics


#endif