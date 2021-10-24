#ifndef NUMERICS_OPTIMIZATION_LMLSQR_HPP
#define NUMERICS_OPTIMIZATION_LMLSQR_HPP

#include "numerics/optimization/optim_base.hpp"
#include "numerics/optimization/fzero.hpp"
#include "numerics/derivatives.hpp"

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

#ifdef NUMERICS_WITH_ARMA
namespace numerics {
namespace optimization
{

namespace __lmlsqr
{

template <typename real>
class lmlsqr_step
{
private:
    real delta;

public:
    lmlsqr_step(const arma::Col<real>& x)
    {
        delta = 0.9*std::max<real>(1.0f, arma::norm(x,"inf"));
    }

    template <class Func, class Jac, typename = typename std::enable_if<std::is_invocable<Jac,arma::Col<real>>::value>::type>
    bool operator()(arma::Col<real>& dx, arma::Col<real>& x, arma::Col<real>& F, Func f, Jac jacobian, const OptimizationOptions<real>& opts)
    {
        arma::Mat<real> J = jacobian(x);
        if (J.has_nan() or (J.n_cols!=x.n_elem) or (J.n_rows!=F.n_elem))
            return false;
        
        arma::Col<real> D;
        arma::Mat<real> U,V;
        bool success = arma::svd_econ(U, D, V, J);
        if (not success)
            return false;
        
        real rho;
        real f0 = 0.5*arma::dot(F,F);
        arma::Col<real> UF = -U.t() * F;

        arma::Col<real> p = V * (UF / D);
        u_long nit = 0;
        real lam;
        while (true) {
            if (nit > 100)
                return false;

            if (arma::norm(p) <= delta) {
                dx = p;
                lam = 0;
            } else {
                auto lam_search = [&](real l) -> real {
                    dx = V * (D / (arma::square(D) + l*l) % UF);
                    return 1/delta - 1/arma::norm(dx);
                };
                lam = std::min<real>(D.min()+std::numeric_limits<real>::epsilon(), 1.0);
                lam = newton_1d(lam_search, lam, 10*std::numeric_limits<real>::epsilon());
                lam = lam * lam;
                dx = V * (D / (arma::square(D)+lam) % UF);
            }

            arma::Col<real> F1 = f(static_cast<arma::Col<real>>(x + dx));
            if (F1.has_nan())
                return false;

            arma::Col<real> Jp = J*dx;
            real f1 = 0.5*arma::dot(F1, F1);
            rho = (f1 - f0) / (arma::dot(F + 0.5*Jp, Jp));
            
            if (rho < 0.25)
                delta = 0.25*arma::norm(dx);
            else if ((rho > 0.75) or (lam == 0))
                delta = std::min<real>(2*delta, 10.0);
            
            if ((f1 < 0.99*f0) or (rho > 0.1))
            {
                F = std::move(F1);
                break;
            }

            ++nit;
        }
        
        return true;
    }
};

}

template <typename real, class Func>
OptimizationResults<arma::Col<real>,real> lmlsqr(arma::Col<real>& x, Func f, const OptimizationOptions<real>& opts=OptimizationOptions<real>())
{
    auto jac = [&f](const arma::Col<real>& z) -> arma::Mat<real>
    {
        return jacobian<real>(std::forward<Func>(f), z, 10*std::sqrt(std::numeric_limits<real>::epsilon()));
    };
    __lmlsqr::lmlsqr_step<real> step(x);
    return __optim_base::gen_solve(x, std::forward<Func>(f), jac, opts, step);
}

template <typename real, class Func, class Jac, typename = typename std::enable_if<std::is_invocable<Jac, arma::Col<real>>::value>::type>
OptimizationResults<arma::Col<real>,real> lmlsqr(arma::Col<real>& x, Func f, Jac jacobian, const OptimizationOptions<real>& opts=OptimizationOptions<real>())
{
    __lmlsqr::lmlsqr_step<real> step(x);
    return __optim_base::gen_solve(x, std::forward<Func>(f), std::forward<Jac>(jacobian), opts, step);
}

}
}
#endif

#endif