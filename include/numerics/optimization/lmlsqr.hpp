#ifndef NUMERICS_OPTIMIZATION_LMLSQR_HPP
#define NUMERICS_OPTIMIZATION_LMLSQR_HPP

#include "numerics/concepts.hpp"
#include "numerics/optimization/optim_base.hpp"
#include "numerics/optimization/fzero.hpp"
#include "numerics/derivatives.hpp"

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

#ifdef NUMERICS_WITH_ARMA
namespace numerics {
    namespace optimization {
        namespace __lmlsqr {
            template <scalar_field_type scalar>
            class lmlsqr_step
            {
            private:
                precision_t<scalar> delta;

            public:
                typedef precision_t<scalar> precision;

                lmlsqr_step(const arma::Col<scalar>& x)
                {
                    delta = 0.9*std::max<precision>(1.0f, arma::norm(x,"inf"));
                }

                template <std::invocable<arma::Col<scalar>> Func, std::invocable<arma::Col<scalar>> Jac>
                bool operator()(arma::Col<scalar>& dx, arma::Col<scalar>& x, arma::Col<scalar>& F, Func f, Jac jacobian, const OptimizationOptions<precision>& opts)
                {
                    arma::Mat<scalar> J = jacobian(x);
                    if (J.has_nan() or (J.n_cols!=x.n_elem) or (J.n_rows!=F.n_elem))
                        return false;
                    
                    arma::Col<precision> D;
                    arma::Mat<scalar> U,V;
                    bool success = arma::svd_econ(U, D, V, J);
                    if (not success)
                        return false;
                    
                    precision rho;
                    precision f0 = precision(0.5)*std::abs(arma::cdot(F,F));
                    arma::Col<scalar> UF = -U.t() * F;

                    arma::Col<scalar> p = V * (UF / D);
                    u_long nit = 0;
                    precision lam;
                    while (true) {
                        if (nit > 100)
                            return false;

                        if (arma::norm(p) <= delta) {
                            dx = p;
                            lam = 0;
                        } else {
                            auto lam_search = [&](precision l) -> precision {
                                dx = V * (D / (arma::square(D) + l*l) % UF);
                                return 1/delta - 1/arma::norm(dx);
                            };
                            lam = std::min<precision>(D.min()+std::numeric_limits<precision>::epsilon(), 1.0);
                            lam = newton_1d(lam_search, lam, 10*std::numeric_limits<precision>::epsilon());
                            lam = lam * lam;
                            dx = V * (D / (arma::square(D)+lam) % UF);
                        }

                        arma::Col<scalar> F1 = f(static_cast<arma::Col<scalar>>(x + dx));
                        if (F1.has_nan())
                            return false;

                        arma::Col<scalar> Jp = J*dx;
                        precision f1 = precision(0.5)*std::abs(arma::dot(F1, F1));
                        scalar dm = arma::cdot(F + scalar(0.5)*Jp, Jp);
                        rho = (f0 - f1) / std::abs(dm);
                        
                        if (rho < 0.25)
                            delta = 0.25*arma::norm(dx);
                        else if ((rho > 0.75) or (lam == 0))
                            delta = std::min<precision>(2*delta, 10.0);
                        
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
        } // namespace __lmlsqr

        // minimizes ||f(x)||^2 for x using the Levenberg-Marquardt algorithm. The
        // vector x should be initialized to some initial guess. The global convergence
        // depends on the quality of this initial guess. This method is well suited for
        // small to medium scale problems since it approximates and factors a dense
        // approximation of the jacobian matrix.
        // see:
        // Manolis I. A.Lourakis "A Brief Description of the Levenberg-Marquardt
        // Algorithm Implemened by levmar", 2005.
        // (2006) Least-Squares Problems. In: Numerical Optimization. Springer Series in
        // Operations Research and Financial Engineering. Springer, New York, NY.
        // https://doi-org.proxy2.cl.msu.edu/10.1007/978-0-387-40065-5_10
        template <scalar_field_type scalar, std::invocable<arma::Col<scalar>> Func>
        OptimizationResults<arma::Col<scalar>> lmlsqr(arma::Col<scalar>& x, Func f, const OptimizationOptions<precision_t<scalar>>& opts={})
        {
            auto jac = [&f](const arma::Col<scalar>& z) -> arma::Mat<scalar>
            {
                return jacobian<scalar>(std::forward<Func>(f), z);
            };
            __lmlsqr::lmlsqr_step<scalar> step(x);
            return __optim_base::gen_solve<arma::Col<scalar>,scalar>(x, std::forward<Func>(f), jac, opts, step);
        }

        // minimizes ||f(x)||^2 for x using the Levenberg-Marquardt algorithm. The
        // jacobian object should be callable with an arma::Col x object and return an
        // arma::Mat object which is the jacobian of f(x). The vector x should be
        // initialized to some initial guess. The global convergence depends on the
        // quality of this initial guess. This method is well suited for small to medium
        // scale problems since it computes and factors the jacobian matrix.
        // see:
        // Manolis I. A.Lourakis "A Brief Description of the Levenberg-Marquardt
        // Algorithm Implemened by levmar", 2005.
        // (2006) Least-Squares Problems. In: Numerical Optimization. Springer Series in
        // Operations Research and Financial Engineering. Springer, New York, NY.
        // https://doi-org.proxy2.cl.msu.edu/10.1007/978-0-387-40065-5_10
        template <scalar_field_type scalar, std::invocable<arma::Col<scalar>> Func, std::invocable<arma::Col<scalar>> Jac>
        OptimizationResults<arma::Col<scalar>> lmlsqr(arma::Col<scalar>& x, Func f, Jac jacobian, const OptimizationOptions<precision_t<scalar>>& opts={})
        {
            __lmlsqr::lmlsqr_step<scalar> step(x);
            return __optim_base::gen_solve<arma::Col<scalar>,scalar>(x, std::forward<Func>(f), std::forward<Jac>(jacobian), opts, step);
        }
    } // namespace optimization
} // namespace numerics
#endif

#endif