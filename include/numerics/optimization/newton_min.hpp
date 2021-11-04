#ifndef NUMERICS_OPTIMIZATION_NEWTON_MIN_HPP
#define NUMERICS_OPTIMIZATION_NEWTON_MIN_HPP

#include "numerics/optimization/optim_base.hpp"
#include "numerics/optimization/descent_cg.hpp"
#include "numerics/optimization/wolfe_step.hpp"

namespace numerics {
namespace optimization
{
namespace __newton_min
{

#ifdef NUMERICS_WITH_ARMA
template <typename real>
inline bool spd_solve_impl(arma::Col<real>& x, const arma::Mat<real>& A, const arma::Col<real>& b)
{
    bool success = arma::solve(x, A, b, arma::solve_opts::likely_sympd + arma::solve_opts::no_approx);
    if ((not success) or (arma::dot(x, A*x) < 0))
        x = b;
    return true;
}

template <typename real>
inline bool spd_solve_impl(arma::Col<real>& x, const arma::SpMat<real>& A, const arma::Col<real>& b)
{
    arma::Col<real> x0 = std::move(x);
    
    arma::superlu_opts opts;
    opts.symmetric = true;
    bool success = arma::spsolve(x, A, b, "superlu", opts);

    if ((not success) or (arma::dot(x, A*x) < 0)) {
        x = std::move(x0);
        success = descent_cg<real>(x, A, b, b.size());
    }

    return success;
}

template <typename real, std::invocable<arma::Col<real>> Grad, std::invocable<arma::Col<real>> Hess>
inline bool step_solve_impl(arma::Col<real>& dx, arma::Col<real>& x, arma::Col<real>& g, Grad df, Hess hessian)
{
    return spd_solve_impl<real>(dx, hessian(x), g);
}
#endif

template <class vec, std::invocable<vec> Grad, std::invocable<vec,vec> Hess, typename real = typename vec::value_type>
inline bool step_solve_impl(vec& dx, vec& x, vec& g, Grad df, Hess hessian)
{
    auto H = [&hessian,&x](const vec& v) -> vec
    {
        return hessian(x, v);
    };
    return descent_cg(dx, H, g, static_cast<real(*)(const vec&,const vec&)>(__optim_base::dot_impl), g.size());
}

template <class vec, std::invocable<vec> Grad, typename real = typename vec::value_type>
inline bool step_solve_impl(vec& dx, vec& x, vec& g, Grad df, int hessian)
{
    auto H = [&g,&df,&x](const vec& v) -> vec
    {
        constexpr real e = std::numeric_limits<real>::epsilon();
        real C = 100 * std::sqrt(e) * std::max<real>(1.0f, __optim_base::norm_impl(x)) / std::max<real>(1.0f, __optim_base::norm_impl(v));
        return (df(x + C*v) + g) / C; // plus because g = -df(x)
    };

    return descent_cg(dx, H, g, static_cast<real(*)(const vec&, const vec&)>(__optim_base::dot_impl), g.size());
}

template <class vec, std::invocable<vec> Func, std::invocable<vec> Grad, class Hess, typename real = typename vec::value_type>
bool newton_step(vec& dx, vec& x, vec& g, Func f, Grad df, Hess hessian, const OptimizationOptions<real>& opts)
{
    bool success = step_solve_impl(dx, x, g, std::forward<Grad>(df), std::forward<Hess>(hessian));

    if (not success)
        return false;
    
    real alpha = wolfe_step(std::forward<Func>(f), std::forward<Grad>(df), x, dx);
    dx *= alpha;

    g = -df(x + dx);
    
    return true;
}

template <class vec, std::invocable<vec> Func, std::invocable<vec> Grad, typename real = typename vec::value_type>
inline bool newton_step_no_hess(vec& dx, vec& x, vec& g, Func f, Grad df, const OptimizationOptions<real>& opts)
{
    return newton_step(dx, x, g, std::forward<Func>(f), std::forward<Grad>(df), int(), opts);
}

}

template <class vec, std::invocable<vec> Func, std::invocable<vec> Grad, typename real = typename vec::value_type>
inline OptimizationResults<real, real> newton_min(vec& x, Func f, Grad df, const OptimizationOptions<real>& opts = {})
{
    return __optim_base::gen_gradient_solve(x, std::forward<Func>(f), std::forward<Grad>(df), int(), opts, __newton_min::newton_step_no_hess<vec, Func, Grad, real>);
}

template <class vec, std::invocable<vec> Func, std::invocable<vec> Grad, class Hess, typename real = typename vec::value_type>
requires std::invocable<Hess, vec, vec>
#ifdef NUMERICS_WITH_ARMA
or std::invocable<Hess, arma::Col<real>>
#endif
inline OptimizationResults<real, real> newton_min(vec& x, Func f, Grad df, Hess hessian, const OptimizationOptions<real>& opts = {})
{
    return __optim_base::gen_gradient_solve(x, std::forward<Func>(f), std::forward<Grad>(df), std::forward<Hess>(hessian), opts, __newton_min::newton_step<vec,Func,Grad,Hess,real>);
}

}
}

#endif