#ifndef NUMERICS_OPTIMIZATION_LBFGS_HPP
#define NUMERICS_OPTIMIZATION_LBFGS_HPP

#include "numerics/optimization/optim_base.hpp"
#include "numerics/optimization/wolfe_step.hpp"
#include "numerics/derivatives.hpp"
#include "numerics/utility.hpp"

namespace numerics {namespace optimization {

#define LBFGS_WOLFE1 1e-4
#define LBFGS_WOLFE2 0.9

namespace __lbfgs
{

template <class vec, typename real = typename vec::value_type>
class lbfgs_step
{
private:
    const u_long steps;
    CycleQueue<vec> S, Y;
    real hdiag;

    void lbfgs_update(vec& p)
    {
        long k = S.size();
        
        if (k > 0) {
            real ro[k] = {0};
            for (long i=0; i < k; ++i)
                ro[i] = 1 / __optim_base::dot_impl(S.at(i), Y.at(i));

            vec q = p;
            real alpha[k] = {0};
            
            for (long i=k-1; i >= 0; --i)
            {
                alpha[i] = ro[i] * __optim_base::dot_impl(S.at(i), q);
                q -= alpha[i] * Y.at(i);
            }

            vec r = hdiag * q;

            for (long i=0; i < k; ++i)
            {
                real beta = ro[i] * __optim_base::dot_impl(Y.at(i), r);
                r += S.at(i) * (alpha[i] - beta);
            }

            p = std::move(r);
        }
    }

public:
    lbfgs_step(u_long steps_to_remember) : steps(steps_to_remember), S(steps_to_remember), Y(steps_to_remember) {}

    template <std::invocable<vec> Func, std::invocable<vec> Grad>
    bool operator()(vec& dx, vec& x, vec& g, Func f, Grad df, const OptimizationOptions<real>& opts)
    {
        vec p = g;
        lbfgs_update(p);

        real alpha = wolfe_step(std::forward<Func>(f), std::forward<Grad>(df), x, p, real(LBFGS_WOLFE1), real(LBFGS_WOLFE2));

        dx = alpha * p;

        vec g1 = -df(x + dx);
        vec y = g - g1;

        g = std::move(g1);

        hdiag = __optim_base::dot_impl(dx, y) / __optim_base::dot_impl(y, y);
        S.push(dx);
        Y.push(std::move(y));

        return true;
    }
};

}

template <class vec, std::invocable<vec> Func, std::invocable<vec> Grad, typename real = typename vec::value_type>
OptimizationResults<real, real> lbfgs(vec& x, Func f, Grad df, const OptimizationOptions<real>& opts = {}, u_long steps=5)
{
    __lbfgs::lbfgs_step<vec, real> step(steps);
    return __optim_base::gen_gradient_solve(x, std::forward<Func>(f), std::forward<Grad>(df), int(), opts, step);
}

}}

#endif