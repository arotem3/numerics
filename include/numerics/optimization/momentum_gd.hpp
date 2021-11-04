#ifndef NUMERICS_OPTIMIZATION_MOMENTUM_DG_HPP
#define NUMERICS_OPTIMIZATION_MOMENTUM_DG_HPP

#include "numerics/optimization/optim_base.hpp"
#include "numerics/optimization/fmin.hpp"

namespace numerics {namespace optimization {

namespace __momentum_gd {

template <class vec, typename real>
class mgd_step
{
private:
    bool line_min;
    bool initialized;
    real damping_param, alpha;

public:
    mgd_step(real a=0.1, real p=0.9) : alpha(a), damping_param(p), line_min(true), initialized(false) {
        if (a <= 0)
            throw std::invalid_argument("momentum_gd() error: require step size (=" + std::to_string(a) + ") > 0");
        if ((p < 0) or (p >= 1))
            throw std::invalid_argument("momentum_gd() error: require 0 <= damping parameter (=" + std::to_string(p) + ") < 1");
    }

    void set_line_search(bool linemin)
    {
        line_min = linemin;
    }

    template <std::invocable<vec> Func, std::invocable<vec> Grad>
    bool operator()(vec& dx, vec& x, vec& g, Func f, Grad df, const OptimizationOptions<real>& opts)
    {
        if (line_min) {
            alpha = fminsearch([&](real a)->real{return f(x + std::abs(a)*g);}, real(0), std::min<real>(1,alpha), real(0.01));
            alpha = std::abs(alpha);
            dx = alpha*g;

            g = -df(static_cast<vec>(x + dx));
        } else {
            if (not initialized) {
                dx = alpha * g;
                initialized = true;  
            }

            g = -df(static_cast<vec>(x + damping_param*dx));
            dx = alpha*g + damping_param * dx;
        }
        return true;
    }
};
}

template<typename real>
struct MomentumOptions : public OptimizationOptions<real>
{
    bool use_line_search = true;
    real alpha = 0.1;
    real damping_param = 0.9;
};

template <class vec, std::invocable<vec> Func, std::invocable<vec> Grad, typename real = typename vec::value_type>
OptimizationResults<real, real> momentum_gd(vec& x, Func f, Grad df, const MomentumOptions<real>& opts = {})
{
    __momentum_gd::mgd_step<vec,real> step(opts.alpha, opts.damping_param);
    step.set_line_search(opts.use_line_search);
    return __optim_base::gen_gradient_solve(x, std::forward<Func>(f), std::forward<Grad>(df), int(), opts, step);
}

}}

#endif