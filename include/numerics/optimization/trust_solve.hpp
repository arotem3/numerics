#ifndef NUMERICS_OPTIMIZATION_trust_solve_HPP
#define NUMERICS_OPTIMIZATION_trust_solve_HPP

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

#include "numerics/concepts.hpp"
#include "numerics/optimization/optim_base.hpp"
#include "numerics/optimization/trust_base.hpp"
#include "numerics/optimization/gmres.hpp"
#include "numerics/optimization/fzero.hpp"
#include "numerics/optimization/fmin.hpp"
#include "numerics/derivatives.hpp"

#include <limits>
#include <type_traits>

namespace numerics {
    namespace optimization {
        namespace __trust_solve {
            template <class vec, scalar_field_type scalar = typename vec::value_type>
            inline precision_t<scalar> norm_squared(const vec& f)
            {
                return std::real(__vmath::dot_impl(f,f));
            }

            template <class vec, scalar_field_type scalar>
            class trust_step
            {
            public:
                typedef precision_t<scalar> precision;

            private:
                precision model_reduction;

                template <std::invocable<vec> Func, std::invocable<vec> Jac>
                std::tuple<bool,vec,vec> search_dirs(const vec& x, const vec& F, Func f, Jac jacobian)
                {
                    auto J = jacobian(x);
                    vec g = J.t() * F;

                    vec p;
                    bool success = __vmath::solve_impl(p, J, F);
                    return std::make_tuple(success, std::move(g), std::move(p));
                }

                template <std::invocable<vec> Func>
                std::tuple<bool,vec,vec> search_dirs(const vec& x, const vec& F, Func f, int dummy_jacobian)
                {
                    auto fnorm = [&f](const vec& z) -> precision
                    {
                        return precision(0.5) * norm_squared(static_cast<vec>(f(z)));
                    };

                    vec g = grad(fnorm, x);

                    auto JacMult = [&F, &f, x](const vec& v) -> vec
                    {
                        return __optim_base::jac_product(x, v, F, std::forward<Func>(f));
                    };

                    vec p = g;

                    precision tol = std::min<precision>(0.5, std::sqrt(f0));
                    bool success = gmres(p, JacMult, F, static_cast<scalar(*)(const vec&,const vec&)>(__vmath::dot_impl), tol, precision(0), 20, x.size()*10);

                    return std::make_tuple(success, std::move(g), std::move(p));
                }

            public:
                precision delta;
                precision delta_max;
                precision f0;

                trust_step() {}

                template <std::invocable<vec> Func, class Jac>
                bool operator()(vec& dx, vec& x, vec& F, Func f, Jac jacobian, const OptimizationOptions<precision>& opts)
                {
                    vec g, p;
                    bool success;
                    std::tie(success, g, p) = search_dirs(x, F, std::forward<Func>(f), std::forward<Jac>(jacobian));

                    // these parameters are used for the two-dimensional subspace problem
                    precision a = norm_squared(__optim_base::jac_product(x,g,F,f)),
                            b = norm_squared(g);
                    precision c, d;

                    if (success) {
                        c = std::real(__vmath::dot_impl(g,p));
                        d = norm_squared(p);
                    }

                    while (true)
                    {
                        bool trust_active = false;
                        if (success) {
                            if (d <= delta*delta)
                                dx = -p;
                            else {
                                precision u0, u1;
                                auto phi = [&](precision tau) -> precision
                                {
                                    tau = tau*tau;
                                    
                                    precision P = (a + tau*c)*(c + tau*d) - std::pow(b + tau*c, 2);
                                    u0 = ((c + tau*d)*b - (b + tau*c)*d) / P;
                                    u1 = ((a + tau*b)*d - (b + tau*c)*b) / P;

                                    precision unorm = b*u0*u0 + 2*c*u0*u1 + d*u1*u1;
                                    return 1/std::sqrt(unorm) - 1/delta;
                                };
                                
                                newton_1d(phi, precision(1.0f));
                                dx = u0*g + u1*p;
                                trust_active = true;
                            }
                        } else { // could not produce newton direction, use Cauchy point
                            precision tau;
                            if (std::pow(b,3) < std::pow(a*delta, 2)) // line minimum inside trust region
                                tau = b / a;
                            else { // line minimum on trust region boundary
                                trust_active = true;
                                tau = delta / std::sqrt(b);
                            }
                            dx = -tau * g;
                        }

                        model_reduction = f0 - norm_squared(static_cast<vec>(F + __optim_base::jac_product(x,dx,F,std::forward<Func>(f))));

                        vec F1 = f(static_cast<vec>(x + dx));
                        precision f1 = norm_squared(F1);

                        if (f1 < f0) {
                            precision rho = (f0 - f1) / model_reduction;
                            if (rho < precision(0.25))
                                delta = precision(0.25) * __vmath::norm_impl(dx);
                            else if ((rho > precision(0.75)) and trust_active)
                                delta = std::min<precision>(2*delta, delta_max);
                            
                            F = std::move(F1);
                            f0 = f1;
                            return true;
                        }
                        else
                            delta /= precision(2.0);

                        if (delta < opts.xtol) {
                            return false;
                        }
                    }

                    return success;
                }

                template <std::invocable<vec> Func>
                inline bool operator()(vec& dx, vec& x, vec& F, Func f, const OptimizationOptions<precision>& opts)
                {
                    return (*this)(dx, x, F, std::forward<Func>(f), int(), opts);
                }
            };
        } // namespace __trust_solve

        // solves f(x) == 0 using the trust region method. The vector x should be
        // initialized with a guess of the solution. The global convergence of the
        // method depends on the quality of the guess. The trust-region subproblem is
        // solved on a two dimensional subspace spanned by the gradient of ||f||^2 and
        // an approximate newton direction. The newton direction is found using
        // restarted gmres.
        // see:
        // (2006) Trust-Region Methods. In: Numerical Optimization. Springer Series in
        // Operations Research and Financial Engineering. Springer, New York, NY.
        // https://doi-org.proxy1.cl.msu.edu/10.1007/978-0-387-40065-5_4
        template <class vec, std::invocable<vec> Func, scalar_field_type scalar = typename vec::value_type>
        inline OptimizationResults<vec> trust_solve(vec& x, Func f, const TrustOptions<precision_t<scalar>>& opts = {})
        {
            __trust_solve::trust_step<vec,scalar> step;
            step.f0 = __trust_solve::norm_squared(f(x));
            
            if (opts.delta <= 0) {
                auto f2 = [&](const vec& x) -> precision_t<scalar>
                {
                    return __trust_solve::norm_squared(static_cast<vec>(f(x)));
                };
                step.delta = initialize_TR_radius(x, static_cast<vec>(-grad(f2, x)), f2, opts.xtol);
            }
            else
                step.delta = opts.delta;
            
            if (opts.delta_max <= 0)
                step.delta = 32*step.delta;
            else
                step.delta_max = opts.delta_max;

            return __optim_base::gen_solve<vec,scalar>(x, std::forward<Func>(f), int(), opts, step);
        }

        #ifdef NUMERICS_WITH_ARMA
        // solves f(x) == 0 using the trust region method. The vector x should be
        // initialized with a guess of the solution. The global convergence of the
        // method depends on the quality of the guess. The object jacobian(x) returns a
        // dense or sparse approximation to the jacobian of f(x). The trust-region
        // subproblem is solved on a two dimensional subspace spanned by the gradient of
        // ||f||^2 and an approximate newton direction. The newton direction is found
        // using a direct solve of the jacobian system.
        // see:
        // (2006) Trust-Region Methods. In: Numerical Optimization. Springer Series in
        // Operations Research and Financial Engineering. Springer, New York, NY.
        // https://doi-org.proxy1.cl.msu.edu/10.1007/978-0-387-40065-5_4
        template <scalar_field_type scalar, std::invocable<arma::Col<scalar>> Func, std::invocable<arma::Col<scalar>> Jac>
        inline OptimizationResults<arma::Col<scalar>> trust_solve(arma::Col<scalar>& x, Func f, Jac jacobian, const TrustOptions<precision_t<scalar>>& opts = {})
        {
            __trust_solve::trust_step<arma::Col<scalar>,scalar> step;
            step.f0 = __trust_solve::norm_squared(f(x));
            
            if (opts.delta <= 0) {
                auto f2 = [&](const arma::Col<scalar>& x) -> precision_t<scalar>
                {
                    return __trust_solve::norm_squared(static_cast<arma::Col<scalar>>(f(x)));
                };
                step.delta = initialize_TR_radius(x, static_cast<arma::Col<scalar>>(-grad(f2, x)), f2, opts.xtol);
            }
            else
                step.delta = opts.delta;
            
            if (opts.delta_max <= 0)
                step.delta = 32*step.delta;
            else
                step.delta_max = opts.delta_max;

            return __optim_base::gen_solve<arma::Col<scalar>,scalar>(x, std::forward<Func>(f), std::forward<Jac>(jacobian), opts, step);
        }
        #endif
    } // namespace optimization
} // namespace numerics
#endif