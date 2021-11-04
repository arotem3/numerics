#ifndef NUMERICS_OPTIMIZATION_BROYDEN_HPP
#define NUMERICS_OPTIMIZATION_BROYDEN_HPP

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

#include "numerics/optimization/optim_base.hpp"
#include "numerics/optimization/fmin.hpp"
#include "numerics/derivatives.hpp"

#ifdef NUMERICS_WITH_ARMA
namespace numerics
{
namespace optimization
{

namespace __broyden
{

template<typename real>
class broyden_step
{
private:
    arma::Mat<real> J;
    bool bad_jacobian;

public:
    void set_jacobian(arma::Mat<real>&& jac)
    {
        J = std::move(jac);
        bad_jacobian = !arma::inv(J, J); // attempt inverse
        if (bad_jacobian) // if singular attempt pseudo-inverse
            bad_jacobian = !arma::pinv(J, J);
    }

    template <class Func, class Jac, typename = typename std::enable_if<std::is_invocable<Jac,arma::Col<real>>::value>::type>
    bool operator()(arma::Col<real>& dx, arma::Col<real>& x, arma::Col<real>& F, Func f, Jac jacobian, const OptimizationOptions<real>& opts)
    {   
        if (bad_jacobian)
            return false;
        
        dx = -(J*F);
        if (dx.has_nan()) {
            set_jacobian( jacobian(x) );
            if (bad_jacobian)
                return false;
            
            dx = -(J*F);
        }

        arma::Col<real> F_old = std::move(F);

        auto line_f = [&dx, &F, &f, &x](real a) -> real
        {
            F = f(static_cast<arma::Col<real>>(x + a*dx));
            return __optim_base::norm_impl(F);
        };

        if (line_f(1.0) > 0.99*__optim_base::norm_impl(F))
        {
            real a = fminbnd(line_f, real(0.0f), real(1.0f), 10*std::sqrt(std::numeric_limits<real>::epsilon()));
            dx *= a;
        }

        if (F.has_nan()) {
            F = std::move(F_old);
            return false;
        }
        
        arma::Col<real> y = F - F_old;
        arma::Col<real> Jy = J * y;
        J += (dx - Jy) * dx.t() * J / arma::dot(dx, Jy);

        return true;
    }
};

}

template <typename real, class Func>
OptimizationResults<arma::Col<real>,real> broyden(arma::Col<real>& x, Func f, const OptimizationOptions<real>& opts=OptimizationOptions<real>())
{
    auto jac = [&f](const arma::Col<real>& z) -> arma::Mat<real>
    {
        return jacobian<real>(std::forward<Func>(f), z, 10*std::sqrt(std::numeric_limits<real>::epsilon()));
    };
    
    __broyden::broyden_step<real> step;
    step.set_jacobian( jac(x) );

    return __optim_base::gen_solve(x, std::forward<Func>(f), jac, opts, step);
}

template <typename real, class Func, class Jac, typename = typename std::enable_if<std::is_invocable<Jac, arma::Col<real>>::value>::type>
OptimizationResults<arma::Col<real>,real> broyden(arma::Col<real>& x, Func f, Jac jacobian, const OptimizationOptions<real>& opts=OptimizationOptions<real>())
{
    __broyden::broyden_step<real> step;
    step.set_jacobian( jacobian(x) );

    return __optim_base::gen_solve(x, std::forward<Func>(f), std::forward<Jac>(jacobian), opts, step);
}

}
}
#endif

#endif