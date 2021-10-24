#ifndef NUMERICS_OPTIMIZATION_TRUST_NEWTON_HPP
#define NUMERICS_OPTIMIZATION_TRUST_NEWTON_HPP

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

#include "numerics/optimization/optim_base.hpp"
#include "numerics/optimization/gmres.hpp"
#include "numerics/optimization/fzero.hpp"
#include "numerics/derivatives.hpp"

#include <limits>
#include <type_traits>

#ifdef NUMERICS_WITH_ARMA
namespace numerics {
namespace optimization
{

namespace __trust_newton
{

template <typename real>
class trust_newton_step
{
private:
    real delta;
    real fh;

public:
    trust_newton_step(const arma::Col<real>& x)
    {
        delta = 0.1 * std::max<real>(1.0f, arma::norm(x,"inf"));
    }

    template <class Func>
    bool operator()(arma::Col<real>& dx, arma::Col<real>& x, arma::Col<real>& F, Func f, const OptimizationOptions<real>& opts)
    {
        real delta_max = 10.0;
        auto fnorm = [&f](const arma::Col<real>& z) -> real
        {
            arma::Col<real> fz = f(z);
            return 0.5*arma::dot(fz,fz);
        };
        arma::Col<real> g = grad(fnorm, x);

        auto JacMult = [&F, f, &x](const arma::Col<real>& v) -> arma::Col<real>
        {
            constexpr real e = std::numeric_limits<real>::epsilon();
            real C = 100 * std::sqrt(e) * std::max<real>(1.0f, __optim_base::norm_impl(x)) / std::max<real>(1.0f, __optim_base::norm_impl(v));
            return (f(x + v*C) - F) / C;
        };
        arma::Col<real> p = dx;
        bool success = gmres(p, JacMult, F, arma::dot<arma::Col<real>,arma::Col<real>>, opts.ftol, opts.ftol, F.size(), 1);
        arma::Col<real> Jg = JacMult(g);

        if (not success)
            return false;
        
        real ff = arma::dot(F,F);
        real gg = arma::dot(g,g);
        real Jg2 = arma::dot(Jg,Jg);
        real pp = arma::dot(p,p);

        arma::Mat<real> A = {
            {ff,gg},
            {gg,Jg2}
        };
        arma::Mat<real> B = {
            {pp,ff},
            {ff,gg}
        };

        arma::Col<real> r = {ff, gg};

        while (true)
        {
            if (arma::norm(p) < delta)
            {
                dx = -p;
                fh = -0.5*arma::dot(g,p);
            }
            else
            {
                arma::Col<real> u;
                auto phi = [this,&u,&A,&B,&r,&p,&g](real l) -> real {
                    u = arma::solve(A + (l*l)*B, -r);
                    real unorm = arma::norm(u[0]*p + u[1]*g);
                    return 1/unorm - 1/delta;
                };
                newton_1d(phi, real(1.0f), 10*std::sqrt(std::numeric_limits<real>::epsilon()));
                dx = u[0]*p + u[1]*g;
                fh = arma::dot(u, 0.5*A*u + r);
            }

            arma::Col<real> F1 = f(x + dx);
            real f1 = arma::dot(F1,F1);

            if (f1 < ff)
            {
                real rho = 0.5*std::abs((ff - f1) / fh);
                if (rho < 0.25)
                    delta = 0.25*arma::norm(dx);
                else if ((rho > 0.75) and (std::abs(arma::norm(dx)-delta) < 10*std::sqrt(std::numeric_limits<real>::epsilon()))) {
                    // approximation is good and dx is on the boundary of the trust region
                    delta = std::min<real>(2*delta, delta_max*arma::norm(x));
                }
                F = std::move(F1);
                success = true;
                break;
            }
            else
                delta /= 2.0;

            if (delta < opts.xtol)
            {
                success = false;
                break;
            }
        }
        return success;
    }

    template <class Func, class Jac, typename = typename std::enable_if<std::is_invocable<Jac,arma::Col<real>>::value>::type>
    bool operator()(arma::Col<real>& dx, arma::Col<real>& x, arma::Col<real>& F, Func f, Jac jacobian, const OptimizationOptions<real>& opts)
    {
        real delta_max = 10.0;

        arma::Mat<real> J = jacobian(x);
        arma::Col<real> g = J.t() * F;
        arma::Col<real> p;
        bool success = arma::solve(p, J, F);
        arma::Col<real> Jg = J * g;

        if (not success)
            return false;
        
        real ff = arma::dot(F,F);
        real gg = arma::dot(g,g);
        real Jg2 = arma::dot(Jg,Jg);
        real pp = arma::dot(p,p);

        arma::Mat<real> A = {
            {ff,gg},
            {gg,Jg2}
        };
        arma::Mat<real> B = {
            {pp,ff},
            {ff,gg}
        };

        arma::Col<real> r = {ff, gg};

        while (true)
        {
            if (arma::norm(p) < delta)
            {
                dx = -p;
                fh = -0.5*arma::dot(g,p);
            }
            else
            {
                arma::Col<real> u;
                auto phi = [this,&u,&A,&B,&r,&p,&g](real l) -> real {
                    u = arma::solve(A + (l*l)*B, -r);
                    real unorm = arma::norm(u[0]*p + u[1]*g);
                    return 1/unorm - 1/delta;
                };
                newton_1d(phi, real(1.0f), 100*std::sqrt(std::numeric_limits<real>::epsilon()));
                dx = u[0]*p + u[1]*g;
                fh = arma::dot(u, 0.5*A*u + r);
            }

            arma::Col<real> F1 = f(x + dx);
            real f1 = arma::dot(F1,F1);

            if (f1 < ff)
            {
                real rho = 0.5*std::abs((ff - f1) / fh);
                if (rho < 0.25)
                    delta = 0.25*arma::norm(dx);
                else if ((rho > 0.75) and (std::abs(arma::norm(dx)-delta) < 10*std::sqrt(std::numeric_limits<real>::epsilon()))) {
                    // approximation is good and dx is on the boundary of the trust region
                    delta = std::min(2*delta, delta_max*arma::norm(x));
                }
                F = std::move(F1);
                success = true;
                break;
            }
            else
                delta /= 2.0;
            
            if (delta < opts.xtol)
            {
                success = false;
                break;
            }
        }
        return success;
    }
};

}

template <typename real, class Func>
OptimizationResults<arma::Col<real>,real> trust_newton(arma::Col<real>& x, Func f, const OptimizationOptions<real>& opts=OptimizationOptions<real>())
{
    __trust_newton::trust_newton_step step(x);
    return __optim_base::gen_solve(x, f, int(), opts, step);
}


template <typename real, class Func, class Jac, typename = typename std::enable_if<std::is_invocable<Jac, arma::Col<real>>::value>::type>
OptimizationResults<arma::Col<real>,real> trust_newton(arma::Col<real>& x, Func f, Jac jacobian, const OptimizationOptions<real>& opts=OptimizationOptions<real>())
{
    __trust_newton::trust_newton_step step(x);
    return __optim_base::gen_solve(x, f, jacobian, opts, step);
}

}
}
#endif

#endif