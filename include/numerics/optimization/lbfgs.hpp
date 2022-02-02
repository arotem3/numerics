#ifndef NUMERICS_OPTIMIZATION_LBFGS_HPP
#define NUMERICS_OPTIMIZATION_LBFGS_HPP

#include <deque>

#include "numerics/optimization/optim_base.hpp"
#include "numerics/optimization/wolfe_step.hpp"
#include "numerics/derivatives.hpp"
#include "numerics/utility.hpp"

namespace numerics {
    namespace optimization {

    #define LBFGS_WOLFE1 1e-4
    #define LBFGS_WOLFE2 0.9

    namespace __lbfgs {
        template <class vec, std::floating_point real = typename vec::value_type>
        class lbfgs_step
        {
        private:
            const u_long steps;
            std::deque<vec> S, Y;
            std::deque<real> rho;
            real hdiag;

            void lbfgs_update(vec& p)
            {
                long k = S.size();
                
                if (k > 0) {
                    vec q = p;
                    real alpha[k] = {0};
                    
                    for (long i=k-1; i >= 0; --i)
                    {
                        alpha[i] = rho.at(i) * __vmath::dot_impl(S.at(i), q);
                        q -= alpha[i] * Y.at(i);
                    }

                    vec r = hdiag * q;

                    for (long i=0; i < k; ++i)
                    {
                        real beta = rho.at(i) * __vmath::dot_impl(Y.at(i), r);
                        r += S.at(i) * (alpha[i] - beta);
                    }

                    p = std::move(r);
                }
            }

        public:
            lbfgs_step(u_long steps_to_remember) : steps(steps_to_remember) {}

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

                hdiag = __vmath::dot_impl(dx, y) / __vmath::dot_impl(y, y);

                if (S.size() == steps) {
                    rho.pop_front();
                    S.pop_front();
                    Y.pop_front();
                }
                rho.push_back(1 / __vmath::dot_impl(dx, y));
                S.push_back(dx);
                Y.push_back(std::move(y));

                return true;
            }
        };
    } // namespace __lbfgs

    template <typename real>
    struct LBFGS_Options : public OptimizationOptions<real>
    {
        u_long steps = 5;
    };

    // minimizes f(x) with respect to x where df(x) is the gradient of f(x) using
    // the limited memory BFGS quasi-Newton method. The vector x should be
    // initialized to some initial guess. The global convergence depends on the
    // quality of this initial guess. This method is well suited to large scale
    // problems since it maintains just a few vectors at each stage of the
    // optimization problem so the memory requirements are O(n), and the cost of
    // each iteration is O(n). To specify the number of previous steps to store for
    // computing the update, the parameter steps can be specified in LBFGS_Options,
    // the default value is 5.
    // see:
    // (2006) Large-Scale Unconstrained Optimization. In: Numerical Optimization.
    // Springer Series in Operations Research and Financial Engineering. Springer,
    // New York, NY. https://doi-org.proxy2.cl.msu.edu/10.1007/978-0-387-40065-5_7
    template <class vec, std::invocable<vec> Func, std::invocable<vec> Grad, std::floating_point real = typename vec::value_type>
    OptimizationResults<real> lbfgs(vec& x, Func f, Grad df, const LBFGS_Options<real>& opts = {})
    {
        __lbfgs::lbfgs_step<vec, real> step(opts.steps);
        return __optim_base::gen_gradient_solve(x, std::forward<Func>(f), std::forward<Grad>(df), int(), opts, step);
    }

    } // namespace optimization
} // namespace numerics

#endif