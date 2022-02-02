#ifndef NUMERICS_OPTIMIZATION_TRUST_BASE_HPP
#define NUMERICS_OPTIMIZATION_TRUST_BASE_HPP

#include "numerics/optimization/optim_base.hpp"

namespace numerics {
    namespace optimization {
        template <std::floating_point real>
        struct TrustOptions : public OptimizationOptions<real>
        {
            real delta = 0;
            real delta_max = 0;
        };

        template <class vec, std::invocable<vec> Func, std::floating_point real = typename vec::value_type>
        real initialize_TR_radius(const vec& x, vec g, Func f, real dxmin)
        {
            real f0 = f(x);
            real delta = __vmath::norm_impl(g);
            g /= delta;

            while (true)
            {
                vec x1 = x + delta*g;
                if (f(x1) < f0) {
                    break;
                }
                else
                    delta *= 0.75;

                if (delta < dxmin) {
                    delta = dxmin;
                    break;
                }
            }

            return delta;
        }
    } // namespace optimization    
} // namespace numerics

#endif