#ifndef NUMERICS_OPTIMIZATION_MOMENTUM_DG_HPP
#define NUMERICS_OPTIMIZATION_MOMENTUM_DG_HPP

#include "numerics/optimization/optim_base.hpp"
#include "numerics/optimization/fmin.hpp"

namespace numerics {
    namespace optimization {
        namespace __momentum_gd {
            template <class vec, std::floating_point real>
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
        } // namespace __momentum_gd

        template<std::floating_point real>
        struct MomentumOptions : public OptimizationOptions<real>
        {
            bool use_line_search = true;
            real alpha = 0.1;
            real damping_param = 0.9;
        };

        // minimizes f(x) with respect to x using gradient descent. The vector x should
        // be initialized to some guess of the solution. The global convergence of this
        // method depends on the quality of this intial guess. The method implements
        // either a line search at each step or using Nesterov momentum accelerated
        // gradient. This is specified in MomentumOptions, if use_line_search = true
        // (the default) then linesearch is used, otherwise momentum is used with step
        // size alpha and an exponential damping parameter damping_param.
        // see:
        // https://en.wikipedia.org/wiki/Gradient_descent
        // Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton. 2013. On the
        // importance of initialization and momentum in deep learning. In Proceedings of
        // the 30th International Conference on International Conference on Machine
        // Learning - Volume 28 (ICML'13). JMLR.org, III–1139–III–1147.
        template <class vec, std::invocable<vec> Func, std::invocable<vec> Grad, std::floating_point real = typename vec::value_type>
        OptimizationResults<real> momentum_gd(vec& x, Func f, Grad df, const MomentumOptions<real>& opts = {})
        {
            __momentum_gd::mgd_step<vec,real> step(opts.alpha, opts.damping_param);
            step.set_line_search(opts.use_line_search);
            return __optim_base::gen_gradient_solve(x, std::forward<Func>(f), std::forward<Grad>(df), int(), opts, step);
        }

        } // namespace optimization
} // namespace numerics

#endif