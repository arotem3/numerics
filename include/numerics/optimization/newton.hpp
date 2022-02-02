#ifndef NUMERICS_OPTIMIZATION_NEWTON_HPP
#define NUMERICS_OPTIMIZATION_NEWTON_HPP

#include "numerics/optimization/optim_base.hpp"
#include "numerics/optimization/fmin.hpp"
#include "numerics/optimization/gmres.hpp"
#include "numerics/derivatives.hpp"

#include <type_traits>

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

namespace numerics {
namespace optimization {
        namespace __newton {
            template <class vec, std::invocable<vec> Func, scalar_field_type scalar=typename vec::value_type>
            inline bool step_solve_impl(vec& dx, vec& x, vec& F, Func f, int jac)
            {
                auto JacMult = [&F,&f,&x](const vec& v) -> vec {
                    return __optim_base::jac_product(x,v,F,std::forward<Func>(f));
                };

                dx = -grad([&](const vec& z)->precision_t<scalar>{vec f1 = f(z); return precision_t<scalar>(0.5)*std::real(__vmath::dot_impl(f1,f1));}, x); // guess the gradient of ||f||^2, any improvement on this point is an improvement over gradient descent

                precision_t<scalar> fnorm = __vmath::norm_impl(F);
                precision_t<scalar> tol = std::min<precision_t<scalar>>(0.5, std::sqrt(fnorm));
                return gmres<vec, decltype(JacMult)>(dx, JacMult, static_cast<vec>(-F), static_cast<scalar(*)(const vec&, const vec&)>(__vmath::dot_impl), tol, precision_t<scalar>(0), 20, 10*F.size());
            }

            #ifdef NUMERICS_WITH_ARMA
            // template <class Func, class Jac, scalar_field_type scalar, typename = typename std::enable_if<std::is_invocable<Jac, arma::Col<real>>::value>::type>
            template <scalar_field_type eT, std::invocable<arma::Col<eT>> Func, std::invocable<arma::Col<eT>> Jac>
            inline bool step_solve_impl(arma::Col<eT>& dx, arma::Col<eT>& x, arma::Col<eT>& F, Func f, Jac jacobian)
            {
                return __vmath::solve_impl<eT>(dx, jacobian(x), -F);
            }
            #endif

            template <scalar_field_type scalar, class vec, std::invocable<vec> Func, class Jac>
            bool newton_step(vec& dx, vec& x, vec& F, Func f, Jac jacobian, const OptimizationOptions<precision_t<scalar>>& opts)
            {
                typedef precision_t<scalar> precision;
                bool success = step_solve_impl(dx, x, F, f, jacobian);

                if (success) {
                    auto line_f = [&F, &x, &f, &dx](precision a) -> precision
                    {
                        F = f(static_cast<vec>(x + a*dx));
                        return std::real(__vmath::dot_impl(F,F));
                    };
                    
                    precision f0 = std::real(__vmath::dot_impl(F,F));
                    if (line_f(precision(1.0)) > precision(0.99)*f0) {
                        precision a = fminbnd(line_f, precision(0.0f), precision(1.0f), precision(0.5)*opts.xtol);
                        dx *= a;
                    }
                }

                return success;
            }

            template <scalar_field_type scalar, class vec, std::invocable<vec> Func>
            inline bool newton_step_no_jac(vec& dx, vec& x, vec& F, Func f, const OptimizationOptions<precision_t<scalar>>& opts)
            {
                return newton_step<scalar, vec>(dx, x, F, std::forward<Func>(f), int(), opts);
            }
        } // namespace __newton


        // solves f(x) == 0 using Newton's method with line search. The vector x should
        // be initialized with a guess of the solution. The global convergence of the
        // method depends on the quality of the guess. The approximate Newton direction
        // is determined using restarted gmres by approximating jacobian vector products
        // via finite differences.
        // see:
        // (2006) Nonlinear Equations. In: Numerical Optimization. Springer Series in
        // Operations Research and Financial Engineering. Springer, New York, NY.
        // https://doi-org.proxy2.cl.msu.edu/10.1007/978-0-387-40065-5_11
        template <class vec, std::invocable<vec> Func, scalar_field_type scalar=typename vec::value_type>
        OptimizationResults<vec> newton(vec& x, Func f, const OptimizationOptions<precision_t<scalar>>& opts={})
        {
            return __optim_base::gen_solve<vec, scalar>(x, std::forward<Func>(f), int(), opts, __newton::newton_step_no_jac<scalar,vec,Func>);
        }

        #ifdef NUMERICS_WITH_ARMA
        // solves f(x) == 0 using Newton's method with line search. The vector x should
        // be initialized with a guess of the solution. The global convergence of the
        // method depends on the quality of the guess. The object jacobian(x) returns
        // the jacobian matrix of f(x) which may be dense or sparse.
        // see:
        // (2006) Nonlinear Equations. In: Numerical Optimization. Springer Series in
        // Operations Research and Financial Engineering. Springer, New York, NY.
        // https://doi-org.proxy2.cl.msu.edu/10.1007/978-0-387-40065-5_11
        template <scalar_field_type eT, std::invocable<arma::Col<eT>> Func, std::invocable<arma::Col<eT>> Jac>
        OptimizationResults<arma::Col<eT>> newton(arma::Col<eT>& x, Func f, Jac jacobian, const OptimizationOptions<precision_t<eT>>& opts={})
        {
            return __optim_base::gen_solve<arma::Col<eT>, eT>(x, std::forward<Func>(f), std::forward<Jac>(jacobian), opts, __newton::newton_step<eT,arma::Col<eT>,Func,Jac>);
        }
        #endif
    } // namespace optimization
} // namespace numerics
#endif