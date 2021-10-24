#ifndef NUMERICS_INTERPOLATION_POLYDER_HPP
#define NUMERICS_INTERPOLATION_POLYDER_HPP

#include <cmath>

namespace numerics
{
/* polyder(p, k) : return the k^th derivative of a polynomial.
 * --- p : polynomial to differentiate.
 * --- k : the derivative order (k = 1 by default, i.e. first derivative). */
template <class Vec, class VecLike>
Vec polyder(const VecLike& p, u_int k = 1) {
    if (p.size() <= k) {
        Vec dp(1);
        dp[0] = 0.0;
        return dp;
    }
    u_long n = p.size() - k;
    Vec dp(n);
    for (u_long i=0; i < n; ++i) {
        dp[i] = std::tgamma(n-i+1.0) / std::tgamma(n-i-k+1.0) * p[i];
    }
    return dp;
}

} // namespace numerics

#endif