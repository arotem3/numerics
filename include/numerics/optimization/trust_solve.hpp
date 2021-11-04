#ifndef NUMERICS_OPTIMIZATION_trust_solve_HPP
#define NUMERICS_OPTIMIZATION_trust_solve_HPP

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

#include "numerics/optimization/optim_base.hpp"
#include "numerics/optimization/gmres.hpp"
#include "numerics/optimization/fzero.hpp"
#include "numerics/optimization/fmin.hpp"
#include "numerics/derivatives.hpp"

#include <limits>
#include <type_traits>

namespace numerics {
namespace optimization
{

namespace __trust_solve
{

template <class vec, typename real = typename vec::value_type>
inline real norm_squared(const vec& f)
{
    return __optim_base::dot_impl(f,f);
}

// dot(W*x, W*y)
template <class vec, typename real = typename vec::value_type>
real weighted_dot(const vec& x, const vec& W, const vec& y)
{
    real wx = 0;
    for (u_long i=0; i < x.size(); ++i)
        wx += x[i] * W[i] * W[i] * y[i];
    return wx;
}

#ifdef NUMERICS_WITH_ARMA
template <typename real>
inline real weighted_dot(const arma::Col<real>& x, const arma::Col<real>& W, const arma::Col<real>& y)
{
    return arma::dot(W%x, W%y);
}
#endif

template <class vec, typename real>
class trust_step
{
private:
    real delta;
    real delta_max;
    real model_reduction;
    vec D;

    template <std::invocable<vec> Func, std::invocable<vec> Jac>
    std::tuple<bool,vec,vec> search_dirs(const vec& dx, const vec& x, const vec& F, Func f, Jac jacobian)
    {
        auto J = jacobian(x);
        vec g = J.t() * F;

        vec p;
        bool success = __optim_base::solve_impl(p, J, F);
        return std::make_tuple(success, std::move(g), std::move(p));
    }

    template <std::invocable<vec> Func>
    std::tuple<bool,vec,vec> search_dirs(const vec& dx, const vec& x, const vec& F, Func f, int dummy_jacobian)
    {
        auto fnorm = [&f](const vec& z) -> real
        {
            return 0.5 * norm_squared(static_cast<vec>(f(z)));
        };

        vec g = grad(fnorm, x);

        auto JacMult = [&F, &f, x](const vec& v) -> vec
        {
            return __optim_base::jac_product(x, v, F, std::forward<Func>(f));
        };

        vec p = dx;
        
        constexpr real t = 100*std::numeric_limits<real>::epsilon();
        bool success = gmres(p, JacMult, F, static_cast<real(*)(const vec&,const vec&)>(__optim_base::dot_impl), real(0.1), t, 20, x.size()*10);
        model_reduction = norm_squared(F) - norm_squared(static_cast<vec>(F - JacMult(p)));
        
        success = success or (model_reduction < 0); // check if p is still a descent direction for ||f||^2

        return std::make_tuple(success, std::move(g), std::move(p));
    }

    template <std::invocable<vec> Func>
    void update_D(const vec& x, Func f)
    {
        real minD = std::numeric_limits<real>::infinity();
        vec D1 = jacobian_diag(std::forward<Func>(f), x);

        for (u_long i=0; i < D.size(); ++i)
        {
            D[i] = std::max<real>(D[i], std::abs(D1[i]));
            minD = std::min<real>(minD, D[i]);
        }

        if (minD != 1)
            D /= minD;
    }

public:
    template <std::invocable<vec> Func>
    trust_step(vec& x, Func f, real dxmin)
    {
        D = 0*x + 1;
        auto fnorm = [&](const vec& y) -> real
        {
            return norm_squared(static_cast<vec>(f(y)));
        };
        
        vec g = grad(fnorm, x);
        
        delta = __optim_base::norm_impl(g);
        g /= delta;

        delta = 0.1 * std::max<real>(1, delta);

        real f0 = fnorm(x);

        // in this loop, we estimate delta for which a gradient descent step
        // decreases the value of the model function. Unlike the optimization
        // version of the trust region method we are not simply minimizing this
        // function, so instead of keeping this step we defer updating x until
        // we compute the newton step.
        while (true)
        {
            vec x1 = x - delta*g;
            if (fnorm(x1) < f0)
                break;
            else
                delta *= 0.75;
            
            if (delta < dxmin)
                break;
        }

        delta_max = 128 * delta;
    }

    template <std::invocable<vec> Func, class Jac>
    bool operator()(vec& dx, vec& x, vec& F, Func f, Jac jacobian, const OptimizationOptions<real>& opts)
    {
        vec g, p;
        bool success;
        std::tie(success, g, p) = search_dirs(dx, x, F, std::forward<Func>(f), std::forward<Jac>(jacobian));

        // these parameters are used for the two-dimensional subspace problem
        real a, b, c, d, beta, alpha, gamma;

        if (success) {
            a = norm_squared(__optim_base::jac_product(x,g,F,f));
            b = norm_squared(g);
            c = norm_squared(p);
            d = norm_squared(F);
            beta = weighted_dot(g,D,g);
            alpha = weighted_dot(g,D,p);
            gamma = weighted_dot(p,D,p);
        }

        while (true)
        {
            bool trust_active = false;
            if (success) {
                if (gamma <= delta*delta)
                    dx = -p;
                else {
                    real u0, u1;
                    auto phi = [&](real tau) -> real
                    {
                        tau = tau*tau;
                        
                        real P = (a + tau*beta)*(d + tau*gamma) - std::pow(b + tau*alpha, 2);
                        u0 = (b*d + tau*b*gamma - b*c - tau*alpha*c) / P;
                        u1 = (-b*b - tau*alpha*b + c*d + tau*gamma*c) / P;

                        real unorm = beta*u0*u0 + 2*alpha*u0*u1 + gamma*u1*u1;
                        return 1/std::sqrt(unorm) - 1/delta;
                    };
                    
                    newton_1d(phi, real(1.0f));
                    dx = u0*g + u1*p;
                    trust_active = true;
                }
            } else {
                dx = g * (-delta / weighted_dot(g, static_cast<vec>(1/D), g));
            }

            model_reduction = d - norm_squared(static_cast<vec>(F + __optim_base::jac_product(x,dx,F,std::forward<Func>(f))));

            vec F1 = f(static_cast<vec>(x + dx));
            real f1 = norm_squared(F1);

            if (f1 < d) {
                real rho = (d - f1) / model_reduction;
                if (rho < 0.25)
                    delta = 0.25 * __optim_base::norm_impl(dx);
                else if ((rho > 0.75) and trust_active)
                    delta = std::min<real>(2*delta, delta_max);
                
                F = std::move(F1);
                return true;
            }
            else
                delta /= 2.0;

            if (delta < opts.xtol) {
                return false;
            }
        }

        return success;
    }

    template <std::invocable<vec> Func>
    inline bool operator()(vec& dx, vec& x, vec& F, Func f, const OptimizationOptions<real>& opts)
    {
        return (*this)(dx, x, F, std::forward<Func>(f), int(), opts);
    }
};

}

template <class vec, std::invocable<vec> Func, typename real = typename vec::value_type>
inline OptimizationResults<vec, real> trust_solve(vec& x, Func f, const OptimizationOptions<real>& opts = {})
{
    __trust_solve::trust_step<vec, real> step(x, std::forward<Func>(f), opts.xtol);
    return __optim_base::gen_solve(x, std::forward<Func>(f), int(), opts, step);
}

#ifdef NUMERICS_WITH_ARMA
template <typename real, std::invocable<arma::Col<real>> Func, std::invocable<arma::Col<real>> Jac>
inline OptimizationResults<arma::Col<real>,real> trust_solve(arma::Col<real>& x, Func f, Jac jacobian, const OptimizationOptions<real>& opts = {})
{
    __trust_solve::trust_step<arma::Col<real>, real> step(x, std::forward<Func>(f), opts.xtol);
    return __optim_base::gen_solve(x, std::forward<Func>(f), std::forward<Jac>(jacobian), opts, step);
}
#endif

}
}
#endif