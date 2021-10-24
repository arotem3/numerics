#ifndef NUMERICS_INTERPOLATION_POLYINT_HPP
#define NUMERICS_INTERPOLATION_POLYINT_HPP

#include <cmath>

namespace numerics
{

/* polyint(p, c) : return the integral of a polynomial.
 * --- p : polynomial to integrate.
 * --- c : integration constant. */
template<class Vec, class VecLike>
Vec polyint(const VecLike& p, double c) {
    u_long n = p.size() + 1;
    Vec ip(n);
    for (u_long i=0; i < n-1; ++i) {
        ip[i] = p[i] / (n-1-i);
    }
    ip[n-1] = c;
    return ip;
}

}

#endif