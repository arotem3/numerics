#ifndef NUMERICS_OPTIMIZATION_TRUST_MIN_HPP
#define NUMERICS_OPTIMIZATION_TRUST_MIN_HPP

#include "numerics/optimization/optim_base.hpp"
#include "numerics/optimization/fzero.hpp"
#include "numerics/derivatives.hpp"

namespace numerics { namespace optimization {

namespace __trust_min {

template <class vec, typename real=typename vec::value_type>
class trust_step
{
private:
    real f0, model_reduction, delta, delta_max;
    vec D;

    // set p to p + tau*d such that ||p + tau*d|| = delta
    inline void to_trust_region(vec& p, const vec& d)
    {
        real a = __optim_base::dot_impl(d,d),
             b = __optim_base::dot_impl(p,d),
             c = __optim_base::dot_impl(p,p) - delta*delta;
        
        real tau = (-b + std::sqrt(b*b - a*c))/a;
        p += tau*d;
    }

    // Steihaug's approximate solution to the trust region sub-problem using
    // conjugate gradient iteration. Returns true if ||p|| == delta
    template <std::invocable<vec> LinOp>
    bool steihaug_cg(vec& p, LinOp B, const vec& g, real tol)
    {
        p = 0*g;
        vec r = g / D;
        vec d;

        real gnorm = __optim_base::norm_impl(g);

        real rho_prev, rho = gnorm*gnorm;
        for (u_long i=0; i < g.size(); ++i)
        {
            if (i == 0)
                d = r;
            else {
                real beta = rho / rho_prev;
                d += beta * d;
            }

            vec Bd = B(static_cast<vec>(d / D)) / D; // inv(D) * B * inv(D) * d
            real dBd = __optim_base::dot_impl(d, Bd);

            if (dBd <= 0) {
                to_trust_region(p, d);
                return true;
            }
            
            real alpha = rho / dBd;
            vec p1 = p + alpha * d;

            if (__optim_base::norm_impl(p1) > delta) {
                to_trust_region(p, d);
                return true;
            }

            p = std::move(p1);
            r -= alpha * Bd;

            rho_prev = rho;
            rho = __optim_base::dot_impl(r, r);
            
            if (std::sqrt(rho) < tol * gnorm)
                return false;
        }
        return false;
    }

    // finds a search step based on the trust-region sub-problem for the current
    // trust-region size delta.
    template <std::invocable<vec> Hess>
    inline bool solve_impl(vec& dx, vec& x, vec& g, Hess hessian)
    {
        bool trust_active = steihaug_cg(dx, std::forward<Hess>(hessian), g, 100*std::sqrt(std::numeric_limits<real>::epsilon()));

        dx /= D;
        model_reduction = -__optim_base::dot_impl(dx, static_cast<vec>(0.5*hessian(dx) - g));

        return trust_active;
    }

    // wrapper for solve_impl when hessian is not provided. Hessian
    // matrix-vector product is estimated by finite differences
    template <std::invocable<vec> Grad>
    inline bool solve_wrapper(vec& dx, vec& x, vec& g, Grad df, int hessian_dummy)
    {
        auto H = [&](const vec& v) -> vec
        {
            constexpr real e = std::numeric_limits<real>::epsilon();
            real C = 100 * std::sqrt(e) * std::max<real>(1.0, __optim_base::norm_impl(x)) / std::max<real>(1.0, __optim_base::norm_impl(v));
            return (df(x + C*v) + g) / C;
        };
        
        return solve_impl(dx, x, g, H);
    }

    // wrapper for solv_impl when hessian matrix-vector product is provided.
    template <std::invocable<vec> Grad, std::invocable<vec, vec> Hess>
    inline bool solve_wrapper(vec& dx, vec& x, vec& g, Grad df, Hess hessian)
    {
        auto H = [&](const vec& v) -> vec
        {
            return hessian(x, v);
        };

        return solve_impl(dx, x, g, H);
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

    // adaptively estimates the trust region scaling D such that ||D*p|| < delta
    template <std::invocable<vec> Func>
    void update_D(const vec& x, Func f)
    {
        real minD = std::numeric_limits<real>::infinity();
        vec D1 = hessian_diag(std::forward<Func>(f), x);
        for (u_long i=0; i < D.size(); ++i)
        {
            D[i] = std::max<real>(std::sqrt(std::abs(D1[i])), D[i]);
            minD = std::min<real>(minD, D[i]);
        }

        if (minD != 1)
            D /= minD;
    }

public:
    template <std::invocable<vec> Func, std::invocable<vec> Grad>
    trust_step(vec& x, Func f, Grad df)
    {
        f0 = f(x);
        vec g = -df(x);
        D = 0*g + 1;
        delta = __optim_base::norm_impl(g);
        g /= delta;

        while (true)
        {
            vec x1 = x + delta*g;
            if (f(x1) < f0) {
                x = std::move(x1);
                break;
            }
            else
                delta *= 0.75;
        }

        delta_max = 128 * delta;
    }

    explicit inline trust_step(real d = 1)
    {
        delta_max = std::numeric_limits<real>::infinity();
        delta = d;
    }

    inline void set_delta_max(real d)
    {
        delta_max = d;
    }

    template <std::invocable<vec> Func, std::invocable<vec> Grad, typename Hess>
    bool operator()(vec& dx, vec& x, vec& g, Func f, Grad df, Hess hessian, const OptimizationOptions<real>& opts)
    {   
        update_D(x, std::forward<Func>(f));
        while (true)
        {
            bool trust_active = solve_wrapper(dx, x, g, df, hessian);
            real f1 = f(x + dx);

            if (f1 < f0) {
                real rho = (f0 - f1) / model_reduction; // model reduction is computed in step_solve_impl
                real dx_norm = __optim_base::norm_impl(dx);
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

}

template <class vec, std::invocable<vec> Func, std::invocable<vec> Grad, typename real = vec::value_type>
inline OptimizationResults<real,real> trust_min(vec& x, Func f, Grad df, const OptimizationOptions<real>& opts = {})
{
    __trust_min::trust_step<vec,real> step(x, std::forward<Func>(f), std::forward<Grad>(df));
    return __optim_base::gen_gradient_solve(x, std::forward<Func>(f), std::forward<Grad>(df), int(), opts, step);
}

template <class vec, std::invocable<vec> Func, std::invocable<vec> Grad, class Hess, typename real = vec::value_type>
inline OptimizationResults<real, real> trust_min(vec& x, Func f, Grad df, Hess hessian, const OptimizationOptions<real>& opts = {})
{
    __trust_min::trust_step<vec, real> step(x, std::forward<Func>(f), std::forward<Grad>(df));
    return __optim_base::gen_gradient_solve(x, std::forward<Func>(f), std::forward<Grad>(df), std::forward<Hess>(hessian), opts, step);
}


}}

#endif