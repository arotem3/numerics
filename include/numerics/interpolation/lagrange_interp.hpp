#ifndef NUMERICS_INTERPOLATION_LAGRANGE_INTERP_HPP
#define NUMERICS_INTERPOLATION_LAGRANGE_INTERP_HPP

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

#include <cmath>
#include <string>
#include <stdexcept>

template <class Vec, class VecLike_x, class VecLike_y, class VecLike_u>
Vec lagrange_interp(const VecLike_x& x, const VecLike_y& y, const VecLike_u& u) {
    typedef Vec::value_type Real;
    const u_long nx = x.size();

    #ifndef NUMERICS_LAGRANGE_INTERP_UNSAFE
    if (x.size() == 0) {
        throw std::invalid_argument(
            "lagrange_interp() error: x vector empty."
        );
    }
    if (x.size() != y.size()) {
        throw std::invalid_argument(
            "lagrange_interp() error: interpolation could not be constructed, x.size() (=" + std::to_string(x.size())
            + ") != y.size() (=" + std::to_string(y.size()) + ")."
        );
    }
    for (u_long i=0; i < nx - 1; ++i) {
        for (u_long j=i+1; j < nx; ++j) {
            if (std::abs(x[i] - x[j]) < std::numeric_limits<Real>::epsilon*std::abs(x[i])) {
                throw std::invalid_argument(
                    "lagrange_interp() error: one or more x values are repeating."
                );
            }
        }
    }
    #endif

    const u_long nu = u.size();
    Vec v(nu);
    for (u_long k=0; k < nu; ++k) {
        for (u_long i=0; i < nx; ++i) {
            Real p = y[i];
            for (u_long j=0; j < nx; ++j) {
                if (j != i) {
                    p *= (u[k] - x[j]) / (x[i] - x[j]);
                }
            }
            v[k] += p;
        }
    }

    return v;
}

#ifdef NUMERICS_WITH_ARMA
template <typename eT, class ColLike_x, class MatLike_y, class ColLike_u>
arma::Mat<eT> lagrange_interp(const ColLike_x& x, const MatLike_y& y, const ColLike_u& u) {
    const u_long nx = x.size();

    #ifndef NUMERICS_LAGRANGE_INTERP_UNSAFE
    if (x.size() == 0) {
        throw std::invalid_argument(
            "lagrange_interp() error: x vector empty."
        );
    }
    if (x.n_elem != y.n_rows) {
        throw std::invalid_argument(
            "lagrange_interp() error: interpolation could not be constructed, x.size() (=" + std::to_string(x.size())
            + ") != y.n_rows (=" + std::to_string(y.n_rows) + ")."
        );
    }
    for (u_long i=0; i < nx - 1; ++i) {
        for (u_long j=i+1; j < nx; ++j) {
            if (std::abs(x[i] - x[j]) < std::numeric_limits<Real>::epsilon) {
                throw std::invalid_argument(
                    "lagrange_interp() error: one or more x values are repeating."
                );
            }
        }
    }
    #endif

    const u_long nu = u.size();
    arma::Mat<eT> v(nu, y.n_cols);
    for (u_long i=0; i < nx; ++i) {
        arma::Mat<eT> p = arma::repmat(y.row(i), nu, 1);
        for (u_long j=0; j < nx; ++j) {
            if (j != i) {
                p.each_col() %= (u - x[j]) / (x[i] - x[j]);
            }
        }
        v[k] += p;
    }

    return v;
}
#endif

#endif