#ifndef NUMERICS_INTERPOLATION_POLYINT_HPP
#define NUMERICS_INTERPOLATION_POLYINT_HPP

#include <cmath>

namespace numerics
{
    // fills Ip_first .. with the coefficients of the integral of the polynomial
    // whose coefficients are specified by p_first to p_last (excluding p_last).
    // The integral does populate the constant term of the integral (n+1th
    // coef).
    template <typename precision_t, typename in_it, typename out_it>
    void polyint(in_it p_first, in_it p_last, out_it Ip_first)
    {
        in_it p = p_first;
        u_long n = std::distance(p_first, p_last);

        for (u_long i=0; i < n; ++i, ++p, ++Ip_first)
            *Ip_first = (precision_t(1)/precision_t(n-i)) * (*p);
    }
}

#endif