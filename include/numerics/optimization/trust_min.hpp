#ifndef NUMERICS_OPTIMIZATION_TRUST_MIN_HPP
#define NUMERICS_OPTIMIZATION_TRUST_MIN_HPP

#include "numerics/optimization/optim_base.hpp"
#include "numerics/optimization/trust_base.hpp"
#include "numerics/optimization/fzero.hpp"
#include "numerics/derivatives.hpp"

#include <iomanip>

namespace numerics {
    namespace optimization {
        namespace __trust_min {
            template <class vec, std::floating_point real=typename vec::value_type>
            class trust_step
            {
            private:
                real model_reduction;

                // set p to p + tau*d such that ||p + tau*d|| = delta
                inline void to_trust_region(vec& p, const vec& d)
                {
                    real a = __vmath::dot_impl(d,d),
                        b = __vmath::dot_impl(p,d),
                        c = __vmath::dot_impl(p,p) - delta*delta;
                    
                    real tau = (-b + std::sqrt(b*b - a*c))/a;
                    p += tau*d;
                }

                // Steihaug's approximate solution to the trust region sub-problem using
                // conjugate gradient iteration. Returns true if ||p|| == delta
                template <std::invocable<vec> LinOp>
                bool steihaug_cg(vec& p, LinOp B, const vec& g)
                {
                    p = 0*g;
                    vec r = -g;
                    vec d = g;
                    real rho = __vmath::dot_impl(r,r), rho_prev;

                    real gnorm = __vmath::norm_impl(g);
                    real tol = std::min<real>(0.5, std::sqrt(gnorm)) * gnorm; // this choice ensures superlinear convergence

                    for (u_long i=0; i < g.size(); ++i)
                    {
                        vec Bd = B(d);
                        real dBd = __vmath::dot_impl(d, Bd);
                        if (dBd <= 0) {
                            to_trust_region(p, d);
                            return true;
                        }

                        real alpha = rho / dBd;
                        vec p1 = p + alpha * d;
                        if (__vmath::norm_impl(p1) >= delta) {
                            to_trust_region(p, d);
                            return true;
                        }
                        p = std::move(p1);

                        r += alpha * Bd;
                        rho_prev = rho;
                        rho = __vmath::dot_impl(r,r);
                        if (std::sqrt(rho) < tol)
                            return false;

                        real beta = rho / rho_prev;
                        d = -r + beta*d;
                    }
                    return false;
                }

                // finds a search step based on the trust-region sub-problem for the current
                // trust-region size delta.
                template <std::invocable<vec> Hess>
                inline bool solve_impl(vec& dx, vec& x, vec& g, Hess hessian)
                {
                    bool trust_active = steihaug_cg(dx, std::forward<Hess>(hessian), g);

                    model_reduction = __vmath::dot_impl(dx, static_cast<vec>(g - 0.5*hessian(dx)));

                    return trust_active;
                }

                // wrapper for solve_impl when hessian is not provided. Hessian
                // matrix-vector product is estimated by finite differences
                template <std::invocable<vec> Grad>
                inline bool solve_wrapper(vec& dx, vec& x, vec& g, Grad df, int hessian_dummy)
                {
                    auto H = [&](const vec& v) -> vec
                    {
                        real C = std::sqrt(std::numeric_limits<real>::epsilon()) * std::max<real>(1.0, __vmath::norm_impl(x)) / std::max<real>(1.0, __vmath::norm_impl(v));
                        return (df(x + C*v) + g) / C;
                    };
                    
                    return solve_impl(dx, x, g, std::ref(H));
                }

                // wrapper for solv_impl when hessian matrix-vector product is provided.
                template <std::invocable<vec> Grad, std::invocable<vec, vec> Hess>
                inline bool solve_wrapper(vec& dx, vec& x, vec& g, Grad df, Hess hessian)
                {
                    auto H = [&](const vec& v) -> vec
                    {
                        return hessian(x, v);
                    };

                    return solve_impl(dx, x, g, std::ref(H));
                }

                // wrapper for solve_impl when hessian matrix function is provided. Since
                // steihaug cg is always used, no matrix factorizations are ever computed so
                // arma is not need, however, this form of the hessian function requires
                // hessian(x) to return an object such that hessian(vec)*vec -> vec
                template <std::invocable<vec> Grad, std::invocable<vec> Hess>
                inline bool solve_wrapper(vec& dx, vec& x, vec& g, Grad df, Hess hessian)
                {
                    auto hess_mat = hessian(x);
                    auto H = [&](const vec& v) -> vec
                    {
                        return hess_mat * v;
                    };
                    
                    return solve_impl(dx, x, g, std::ref(H));
                }

            public:
                real delta, delta_max, f0;

                trust_step() {}

                template <std::invocable<vec> Func, std::invocable<vec> Grad, typename Hess>
                bool operator()(vec& dx, vec& x, vec& g, Func f, Grad df, Hess hessian, const OptimizationOptions<real>& opts)
                {
                    while (true)
                    {
                        bool trust_active = solve_wrapper(dx, x, g, df, hessian);
                        real f1 = f(x + dx);
                        if (f1 < f0) {
                            real rho = (f0 - f1) / model_reduction; // model reduction is computed in step_solve_impl
                            real dx_norm = __vmath::norm_impl(dx);
                            if (rho < 0.25)
                                delta = 0.25*dx_norm;
                            else if ((rho > 0.75) and trust_active)
                                delta = std::min<real>(2*delta, delta_max);
                            f0 = f1;
                            g = -df(static_cast<vec>(x + dx));
                            return true;
                        }
                        else
                            delta *= 0.5;

                        if (delta < opts.xtol) {
                            dx *= 0;
                            return true;
                        }
                    }
                }

                template <std::invocable<vec> Func, std::invocable<vec> Grad>
                inline bool operator()(vec& dx, vec& x, vec& g, Func f, Grad df, const OptimizationOptions<real>& opts)
                {
                    return (*this)(dx, x, g, std::forward<Func>(f), std::forward<Grad>(df), int(), opts);
                }
            };
        } // namespace __trust_min

        // minimizes f(x) using the trust region algorithm. The object df(x) is the
        // gradient of f(x). The vector x should be initialized with a guess of the
        // solution. The global convergence of the method depends on the quality of the
        // guess. The trust-region subproblem is solved using Steihaug's conjugate
        // gradient method. 
        // see:
        // (2006) Trust-Region Methods. In: Numerical Optimization. Springer Series in
        // Operations Research and Financial Engineering. Springer, New York, NY.
        // https://doi-org.proxy1.cl.msu.edu/10.1007/978-0-387-40065-5_4
        template <class vec, std::invocable<vec> Func, std::invocable<vec> Grad, std::floating_point real = vec::value_type>
        inline OptimizationResults<real> trust_min(vec& x, Func f, Grad df, const TrustOptions<real>& opts = {})
        {
            __trust_min::trust_step<vec,real> step;
            
            if (opts.delta <= 0)
                step.delta = initialize_TR_radius(x, static_cast<vec>(-df(x)), std::forward<Func>(f), opts.xtol);
            else
                step.delta = opts.delta;
            
            if (opts.delta_max <= 0)
                step.delta_max = 20*step.delta;
            else
                step.delta_max = opts.delta_max;

            step.f0 = f(x);

            return __optim_base::gen_gradient_solve(x, std::forward<Func>(f), std::forward<Grad>(df), int(), opts, step);
        }

        // minimizes f(x) using the trust region algorithm. The object df(x) is the
        // gradient of f(x). The vector x should be initialized with a guess of the
        // solution. The global convergence of the method depends on the quality of the
        // guess. The Hess class should be an invocable object that either produces the
        // product H(x)*v when called with x and v, where H(x) is the hessian of f(x),
        // or produces the jacobian matrix when called with just x. The latter option is
        // only compatible with armadillo types. The jacobian matrix can be sparse or
        // dense. The trust-region subproblem is solved using Steihaug's conjugate
        // gradient method.
        // see:
        // (2006) Trust-Region Methods. In: Numerical Optimization. Springer Series in
        // Operations Research and Financial Engineering. Springer, New York, NY.
        // https://doi-org.proxy1.cl.msu.edu/10.1007/978-0-387-40065-5_4
        template <class vec, std::invocable<vec> Func, std::invocable<vec> Grad, class Hess, std::floating_point real = vec::value_type>
        inline OptimizationResults<real> trust_min(vec& x, Func f, Grad df, Hess hessian, const TrustOptions<real>& opts = {})
        {
            __trust_min::trust_step<vec,real> step;
            
            if (opts.delta <= 0)
                step.delta = initialize_TR_radius(x, static_cast<vec>(-df(x)), std::forward<Func>(f), opts.xtol);
            else
                step.delta = opts.delta;
            
            if (opts.delta_max <= 0)
                step.delta_max = 20*step.delta;
            else
                step.delta_max = opts.delta_max;

            step.f0 = f(x);
                
            return __optim_base::gen_gradient_solve(x, std::forward<Func>(f), std::forward<Grad>(df), std::forward<Hess>(hessian), opts, step);
        }
    } // namespace optimization
} // namespace numerics

#endif