#ifndef NUMERICS_OPTIMIZATION_NEWTON_HPP
#define NUMERICS_OPTIMIZATION_NEWTON_HPP

#include "numerics/optimization/optim_base.hpp"
#include "numerics/optimization/fmin.hpp"
#include "numerics/optimization/gmres.hpp"

#include <type_traits>

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

namespace numerics
{
namespace optimization
{

namespace __newton
{

template <class vec, class Func, typename real=typename vec::value_type>
inline bool step_solve_impl(vec& dx, vec& x, vec& F, Func f, int jac, real tol)
{
    auto JacMult = [&F,&f,&x](const vec& v) -> vec {
        constexpr real e = std::numeric_limits<real>::epsilon();
        real C = 100 * std::sqrt(e) * std::max<real>(1.0f, __optim_base::norm_impl(x)) / std::max<real>(1.0f, __optim_base::norm_impl(v));
        return (f(static_cast<vec>(x + C*v)) - F) / C;
    };

    return gmres<vec, decltype(JacMult)>(dx, JacMult, static_cast<vec>(-F), static_cast<real(*)(const vec&, const vec&)>(__optim_base::dot_impl), tol, tol, F.size(), 1);
}

#ifdef NUMERICS_WITH_ARMA
template <class Func, class Jac, typename real, typename = typename std::enable_if<std::is_invocable<Jac, arma::Col<real>>::value>::type>
inline bool step_solve_impl(arma::Col<real>& dx, arma::Col<real>& x, arma::Col<real>& F, Func f, Jac jacobian, real tol)
{
    return __optim_base::solve_impl<real>(dx, jacobian(x), -F);
}
#endif

template <class Jac, class vec, class Func, typename real>
bool newton_step(vec& dx, vec& x, vec& F, Func f, Jac jacobian, const OptimizationOptions<real>& opts)
{
    bool success = step_solve_impl(dx, x, F, f, jacobian, opts.ftol);

    if (success) {
        auto line_f = [&F, &x, &f, &dx](real a) -> real
        {
            F = f(static_cast<vec>(x + a*dx));
            return __optim_base::norm_impl(F);
        };

        if (line_f(1.0) > 0.99*__optim_base::norm_impl(F)) {
            real a = fminbnd(line_f, real(0.0f), real(1.0f), 10*std::sqrt(std::numeric_limits<real>::epsilon()));
            dx *= a;
        }
    }

    return success;
}

template <class vec, class Func, typename real>
inline bool newton_step_no_jac(vec& dx, vec& x, vec& F, Func f, const OptimizationOptions<real>& opts)
{
    return newton_step(dx, x, F, std::forward<Func>(f), int(), opts);
}

}

template <class vec, class Func, typename real=typename vec::value_type>
OptimizationResults<vec,real> newton(vec& x, Func f, const OptimizationOptions<real>& opts=OptimizationOptions<real>())
{
    return __optim_base::gen_solve(x, std::forward<Func>(f), int(), opts, __newton::newton_step_no_jac<vec,Func,real>);
}

#ifdef NUMERICS_WITH_ARMA
template <typename real, class Func, class Jac, typename = typename std::enable_if<std::is_invocable<Jac, arma::Col<real>>::value>::type>
OptimizationResults<arma::Col<real>,real> newton(arma::Col<real>& x, Func f, Jac jacobian, const OptimizationOptions<real>& opts=OptimizationOptions<real>())
{
    return __optim_base::gen_solve(x, std::forward<Func>(f), std::forward<Jac>(jacobian), opts, __newton::newton_step<Jac,arma::Col<real>,Func,real>);
}
#endif

}
}
#endif