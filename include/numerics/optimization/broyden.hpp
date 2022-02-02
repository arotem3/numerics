#ifndef NUMERICS_OPTIMIZATION_BROYDEN_HPP
#define NUMERICS_OPTIMIZATION_BROYDEN_HPP

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

#include "numerics/optimization/optim_base.hpp"
#include "numerics/optimization/fmin.hpp"
#include "numerics/derivatives.hpp"

#ifdef NUMERICS_WITH_ARMA
namespace numerics {
    namespace optimization {
        namespace __broyden {
            template<scalar_field_type eT>
            class broyden_step
            {
            private:
                arma::Mat<eT> J;
                bool bad_jacobian;

            public:
                typedef precision_t<eT> precision;

                void set_jacobian(arma::Mat<eT>&& jac)
                {
                    J = std::move(jac);
                    bad_jacobian = !arma::inv(J, J); // attempt inverse
                    if (bad_jacobian) // if singular attempt pseudo-inverse
                        bad_jacobian = !arma::pinv(J, J);
                }

                template <std::invocable<arma::Col<eT>> Func, std::invocable<arma::Col<eT>> Jac>
                bool operator()(arma::Col<eT>& dx, arma::Col<eT>& x, arma::Col<eT>& F, Func f, Jac jacobian, const OptimizationOptions<precision_t<eT>>& opts)
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

                    arma::Col<eT> F_old = std::move(F);

                    auto line_f = [&dx, &F, &f, &x](precision a) -> precision
                    {
                        F = f(static_cast<arma::Col<eT>>(x + eT(a)*dx));
                        return __vmath::norm_impl(F);
                    };

                    if (line_f(precision(1.0)) > precision(0.99)*__vmath::norm_impl(F))
                    {
                        precision tol = std::max<precision>(std::sqrt(opts.xtol) / std::max<precision>(1, std::sqrt(__vmath::norm_impl(dx))), std::sqrt(std::numeric_limits<precision>::epsilon()));
                        eT a = fminbnd(line_f, precision(0.0f), precision(1.0f), tol);
                        dx *= a;
                    }

                    if (F.has_nan()) {
                        F = std::move(F_old);
                        return false;
                    }
                    
                    arma::Col<eT> y = F - F_old;
                    arma::Col<eT> Jy = J * y;
                    J += (dx - Jy) * dx.t() * J / arma::cdot(dx, Jy);

                    return true;
                }
            };
        } // namespace __broyden

        // solves f(x) == 0 for x using Broyden's quasi-Newton algorithm with line
        // search. The parameter x should be initialized to some value guess of the
        // solution. The global convergence of this method depends on the quality of
        // this approximation. This method is suitable for small to medium scale
        // problems because it maintains an approximation of the inverse of the jacobian
        // matrix. The approximation is initialized using the finite differences.
        // support complex variable - complex valued function so long as it is analytic;
        // for complex variable problems initialize x with complex values.
        // see:
        // (2006) Nonlinear Equations. In: Numerical Optimization. Springer Series in
        // Operations Research and Financial Engineering. Springer, New York, NY.
        // https://doi-org.proxy2.cl.msu.edu/10.1007/978-0-387-40065-5_11
        template <scalar_field_type eT, std::invocable<arma::Col<eT>> Func>
        OptimizationResults<arma::Col<eT>> broyden(arma::Col<eT>& x, Func f, const OptimizationOptions<precision_t<eT>>& opts={})
        {
            auto jac = [&f](const arma::Col<eT>& z) -> arma::Mat<eT>
            {
                return jacobian<eT>(std::forward<Func>(f), z, 10*std::sqrt(std::numeric_limits<precision_t<eT>>::epsilon()));
            };
            
            __broyden::broyden_step<eT> step;
            step.set_jacobian( jac(x) );

            return __optim_base::gen_solve<arma::Col<eT>, eT>(x, std::forward<Func>(f), jac, opts, step);
        }

        // solves f(x) == 0 for x where jacobian(x) is the jacobian of f(x). using
        // Broyden's quasi-Newton algorithm with line search. The parameter x should be
        // initialized to some value guess of the solution. The global convergence of
        // this method depends on the quality of this approximation. This method is
        // suitable for small to medium scale problems because it maintains an
        // approximation of the inverse of the jacobian matrix. The approximation is
        // initialized using the provided jacobian function. support complex variable -
        // complex valued function so long as it is analytic; for complex variable
        // problems initialize x with complex values.
        // see:
        // (2006) Nonlinear Equations. In: Numerical Optimization. Springer Series in
        // Operations Research and Financial Engineering. Springer, New York, NY.
        // https://doi-org.proxy2.cl.msu.edu/10.1007/978-0-387-40065-5_11
        template <scalar_field_type eT, std::invocable<arma::Col<eT>> Func, std::invocable<arma::Col<eT>> Jac>
        OptimizationResults<arma::Col<eT>> broyden(arma::Col<eT>& x, Func f, Jac jacobian, const OptimizationOptions<precision_t<eT>>& opts={})
        {
            __broyden::broyden_step<eT> step;
            step.set_jacobian( jacobian(x) );

            return __optim_base::gen_solve<arma::Col<eT>, eT>(x, std::forward<Func>(f), std::forward<Jac>(jacobian), opts, step);
        }
    } // namespace optimization
} // namespace numerics
#endif

#endif