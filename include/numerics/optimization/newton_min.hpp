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

template <class vec, class Grad, class Hess, typename real = typename vec::value_type, typename = typename std::enable_if<std::is_invocable<Hess, vec>::value>::type>
inline bool step_solve_impl(vec& dx, vec& x, vec& g, Grad df, Hess hessian)
{
    return spd_solve_impl<real>(dx, hessian(x), g);
}
#endif

template <class vec, class Grad, class Hess, typename real = typename vec::value_type, typename = typename std::enable_if<std::is_invocable<Hess, vec, vec>::value>::type>
inline bool step_solve_impl(vec& dx, vec& x, vec& g, Grad df, Hess hessian)
{
    auto H = [&hessian](const vec& v) -> vec
    {
        return H(x, v);
    };
    return descent_cg(dx, H, g, static_cast<real(*)(const vec&,const vec&)>(__optim_base::dot_impl), g.size());
}

template <class vec, class Grad, typename real = typename vec::value_type>
inline bool step_solve_impl(vec& dx, vec& x, vec& g, Grad df, int hessian)
{
    auto HessMult = [&g,&df,&x](const vec& v) -> vec {
        constexpr real e = std::numeric_limits<real>::epsilon();
        real C = 100 * std::sqrt(e) * std::max<real>(1.0f, __optim_base::norm_impl(x)) / std::max<real>(1.0f, __optim_base::norm_impl(v));
        return (df(x + C*v) + g) / C; // plus because g = -df(x)
    };

    return descent_cg(dx, HessMult, g, static_cast<real(*)(const vec&, const vec&)>(__optim_base::dot_impl), g.size());
}

template <class vec, class Func, class Grad, class Hess, typename real>
bool newton_step(vec& dx, vec& x, vec& g, Func f, Grad df, Hess hessian, const OptimizationOptions<real>& opts)
{
    g = -df(x);
    bool success = step_solve_impl(dx, x, g, std::forward<Grad>(df), std::forward<Hess>(hessian));

    if (not success)
        return false;
    
    real alpha = wolfe_step(std::forward<Func>(f), std::forward<Grad>(df), x, dx);
    dx *= alpha;

    return true;
}

template <class vec, class Func, class Grad, typename real>
inline bool newton_step_no_hess(vec& dx, vec& x, vec& g, Func f, Grad df, const OptimizationOptions<real>& opts)
{
    return newton_step<LinOp>(dx, x, g, std::forward<Func>(f), std::forward<Grad>(df), int(), opts);
}

}

template <class vec, class Func, class Grad, typename real = typename vec::value_type>
inline OptimizationResults<real, real> newton_min(vec& x, Func f, Grad df, const OptimizationOptions<real>& opts = {})
{
    return __optim_base::gen_gradient_solve(x, std::forward<Func>(f), std::forward<Grad>(df), int(), opts, __newton_min::newton_step_no_hess<vec, Func, Grad, real>);
}

template <class vec, class Func, class Grad, class Hess, typename real = typename vec::value_type, typename = typename std::enable_if<std::is_invocable<Hess, vec>::value or std::is_invocable<Hess, vec, vec>::value>::type>
inline OptimizationResults<real, real> newton_min(vec& x, Func f, Grad df, Hess hessian, const OptimizationOptions<real>& opts = {})
{
    return __optim_base::gen_gradient_solve(x, std::forward<Func>(f), std::forward<Grad>(df), std::forward<Hess>(hessian), opts, __newton_min::newton_step<vec,Func,Grad,Hess,real>);
}

}
}

#endif