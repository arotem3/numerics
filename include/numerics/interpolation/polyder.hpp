#ifndef NUMERICS_INTERPOLATION_POLYDER_HPP
#define NUMERICS_INTERPOLATION_POLYDER_HPP

#include <cmath>

namespace numerics
{
    // fills dp_first.. with the k-th derivative of a polynomial with
    // coefficients specified by p_first to p_last (excluding p_last).
    // precision_t should be specified as either float or double to indicate the
    // precision of the coefficients.
    template <std::floating_point precision_t, typename in_it, typename out_it>
    void polyder(in_it p_first, in_it p_last, out_it dp_first, int k=1)
    {
        in_it pi = p_first;
        precision_t n = std::distance(p_first, p_last);
        for (precision_t i=0; i < n-k; ++i, ++pi, ++dp_first)
            *dp_first = (std::tgamma(n-i) / std::tgamma(n-i-k)) * (*pi);
    }
} // namespace numerics

#endif