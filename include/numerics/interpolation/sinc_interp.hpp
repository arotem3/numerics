#ifndef NUMERICS_INTERPOLATION_SINC_HPP
#define NUMERICS_INTERPOLATION_SINC_HPP

#include <cmath>
#include <string>
#include <stdexcept>

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

namespace numerics
{
template <class Vec, class VecLike_x, class VecLike_y, class VecLike_u>
Vec sinc_interp(const VecLike_x& x, const VecLike_y& y, const VecLike_u& u) {
    typedef Vec::value_type Real;

    #ifdef NUMERICS_WITH_ARMA
    using sinc = arma::sinc;
    #else
    auto sinc = [](const Real& z) -> Real {
        if (z == 0) return 1;
        else return std::sin(z) / z;
    };
    #endif

    #ifndef NUMERICS_SINC_INTERP_UNSAFE
    if (x.size() == 0) {
        throw std::invalid_argument(
            "sinc_interp() error: x vector empty."
        );
    }
    if (x.size() != y.size()) {
        throw std::invalid_argument(
            "sinc_interp() error: interpolation could not be constructed, x.size() (=" + std::to_string(x.size())
            + ") != y.size() (=" + std::to_string(y.size()) + ")."
        );
    }
    #endif

    const u_long nx = x.size();
    const u_long nu = u.size();
    Real h = x[1] - x[0];

    #ifndef NUMERICS_SINC_INTERP_UNSAFE
    if (h < 0) {
        throw std::invalid_argument(
            "sinc_interp() error: x not sorted."
        );
    }
    for (u_long i=2; i < nx; ++i) {
        if (x[i] - x[i-1] < 0) {
            throw std::invalid_argument(
                "sinc_interp() error: x not sorted."
            );
        }
        if (std::abs(x[i] - x[i-1] - h) < std::numeric_limits<Real>::epsilon) {
            throw std::invalid_argument(
                "sinc_interp() error: x must be uniformly spaced."
            )
        }
    }
    #endif

    Vec v(nu);
    std::fill_n(v.begin(), nu, 0);
    for (u_long k=0; k < nu; ++k) {
        for (u_long i=0; i < nx; ++i) {
            v[k] += y[i] * sinc((u[k] - x[i]) / h);
        }
    }
    return v;
}

#ifdef NUMERICS_WITH_ARMA
template <typename eT, class ColLike_x, class MatLike_y, class VecLike_u>
arma::Mat<eT> sinc_interp(const ColLike_x& x, const MatLike_y& y, const VecLike_u& u) {
    #ifndef NUMERICS_SINC_INTERP_UNSAFE
    if (x.size() == 0) {
        throw std::invalid_argument(
            "sinc_interp() error: x vector empty."
        );
    }
    if (x.size() != y.n_rows) {
        throw std::invalid_argument(
            "sinc_interp() error: interpolation could not be constructed, x.size() (=" + std::to_string(x.size())
            + ") != y.n_rows (=" + std::to_string(y.n_rows) + ")."
        );
    }
    if (not x.is_sorted()) {
        throw std::invalid_argument(
            "sinc_interp() error: x not sorted."
        );
    }
    #endif

    const u_long nx = x.size();
    const u_long nu = u.size();
    Real h = x[1] - x[0];

    #ifndef NUMERICS_SINC_INTERP_UNSAFE
    if (not arma::all(arma::abs(arma::diff(x) - h) < std::numeric_limits<eT>::epsilon*arma::abs(x)) {
        throw std::invalid_argument(
            "sinc_interp() error: x must be uniformly spaced."
        );
    }
    #endif

    arma::Mat<eT> v = arma::zeros(nu, y.n_cols);
    for (u_long i=0; i < nx; ++i) {
        v +=  arma::sinc((u - x[i]) / h).as_col() * y.row(i);
    }
    return v;
}
#endif
}

#endif