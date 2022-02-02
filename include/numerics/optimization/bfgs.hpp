#ifndef NUMERICS_OPTIMIZATION_BFGS_HPP
#define NUMERICS_OPTIMIZATION_BFGS_HPP

#include "numerics/optimization/optim_base.hpp"
#include "numerics/optimization/wolfe_step.hpp"
#include "numerics/derivatives.hpp"

#ifdef NUMERICS_WITH_ARMA
namespace numerics {
    namespace optimization
    {

    #define BFGS_WOLFE1 1e-4
    #define BFGS_WOLFE2 0.9

        namespace __bfgs {         
            template <std::floating_point real>
            bool inv_sympd(arma::Mat<real>& A)
            {
                // arma does not use cholesky factorization for tiny matrices during call to
                // inv_sympd leading us to compute the inverse of a potentially indef matrix
                // which would cause bfgs to behave unexpectedly.
                if (A.n_rows > 10)
                    return arma::inv_sympd(A, A);
                else {
                    bool chol = arma::chol(A, A, "upper");
                    if (chol) {
                        A = arma::inv(arma::trimatu(A));
                        A = A * A.t();
                        return true;
                    }
                    else
                        return false;
                }
            }

            template <std::floating_point real>
            class bfgs_step
            {
            private:
                arma::Mat<real> H;

            public:
                bfgs_step() {}

                inline void set_hessian(arma::Mat<real>&& hessian)
                {
                    H = std::move(hessian);
                }

                template <std::invocable<arma::Col<real>> Func, std::invocable<arma::Col<real>> Grad, std::invocable<arma::Col<real>> Hess>
                bool operator()(arma::Col<real>& dx, arma::Col<real>& x, arma::Col<real>& g, Func f, Grad df, Hess hessian, const OptimizationOptions<real>& opts)
                {        
                    if (g.has_nan())
                        return false;

                    dx = H*g;

                    if (dx.has_nan()) {
                        H = hessian(x);
                        bool chol = inv_sympd(H);
                        if (not chol)
                            H = arma::eye<arma::Mat<real>>(x.size(), x.size());
                        
                        dx = H*g;
                        
                        if (dx.has_nan())
                            return false;
                    }

                    real alpha = wolfe_step(std::forward<Func>(f), std::forward<Grad>(df), x, dx, real(BFGS_WOLFE1), real(BFGS_WOLFE2));

                    // std::cout << alpha << std::endl;

                    dx *= alpha;

                    arma::Col<real> g1 = -df(x + dx);
                    if (g1.has_nan())
                        return false;

                    arma::Col<real> y = g - g1;
                    arma::Col<real> Hy = H*y;

                    real sy = arma::dot(dx, y);
                    real yHy = arma::dot(y, Hy);

                    auto sHy = dx*Hy.t() / sy;
                    H += (sy + yHy) / (std::pow(sy,2)) * dx*dx.t() - sHy - sHy.t();

                    g = std::move(g1);

                    return true;
                }
            };
        } // namespace __bfgs

        // minimizes f(x) where df(x) is the gradient of f(x) using the
        // Broyden-Fletcher-Goldfarb-Shanno quasi-newton algorithm. The parameter x
        // should be initialized to some value guess of the solution. The global
        // convergence of this method depends on the quality of this approximation. This
        // method is best suited for small and medium scale problems because it
        // maintains a dense approximation of the inverse of the hessian of f. The
        // solver attempts to initialize the hessian using finite differences, but if
        // the hessian is found to be indefinite, it is initialized by the identity
        // matrix instead. In all iterations of the algorithm, the approximate Hessian
        // maintains positive definiteness making it very suitable for convex problem.
        // see:
        // (2006) Quasi-Newton Methods. In: Numerical Optimization. Springer Series in
        // Operations Research and Financial Engineering. Springer, New York, NY.
        // https://doi-org.proxy2.cl.msu.edu/10.1007/978-0-387-40065-5_6
        template <std::floating_point real, std::invocable<arma::Col<real>> Func, std::invocable<arma::Col<real>> Grad>
        OptimizationResults<real> bfgs(arma::Col<real>& x, Func f, Grad df, const OptimizationOptions<real>& opts = {})
        {
            auto hessian = [&df](const arma::Col<real>& z) -> arma::Mat<real>
            {
                arma::Mat<real> h = jacobian(std::forward<Grad>(df), z);
                return 0.5 * (h + h.t());
            };
            arma::Mat<real> H = hessian(x);
            bool chol = __bfgs::inv_sympd(H);
            
            __bfgs::bfgs_step<real> step;
            if (chol)
                step.set_hessian(std::move(H));
            else
                step.set_hessian(arma::eye<arma::Mat<real>>(x.size(), x.size()));

            return __optim_base::gen_gradient_solve(x, std::forward<Func>(f), std::forward<Grad>(df), hessian, opts, step);
        }

        // minimizes f(x) where df(x) is the gradient of f(x), and hessian(x) is the
        // hessian of f(x) using the Broyden-Fletcher-Goldfarb-Shanno quasi-newton
        // algorithm. The parameter x should be initialized to some value guess of the
        // solution. The global convergence of this method depends on the quality of
        // this approximation. This method is best suited for small and medium scale
        // problems because it maintains a dense approximation of the inverse of the
        // hessian of f. The solver attempts to initialize the hessian using the
        // provided hessian, but if the hessian is found to be indefinite, it is
        // initialized by the identity matrix instead. In all iterations of the
        // algorithm, the approximate Hessian maintains positive definiteness making it
        // very suitable for convex problem.
        // see:
        // (2006) Quasi-Newton Methods. In: Numerical Optimization. Springer Series in
        // Operations Research and Financial Engineering. Springer, New York, NY.
        // https://doi-org.proxy2.cl.msu.edu/10.1007/978-0-387-40065-5_6
        template <std::floating_point real, std::invocable<arma::Col<real>> Func, std::invocable<arma::Col<real>> Grad, std::invocable<arma::Col<real>> Hess>
        OptimizationResults<real> bfgs(arma::Col<real>& x, Func f, Grad df, Hess hessian, const OptimizationOptions<real>& opts = {})
        {
            arma::Mat<real> H = hessian(x);
            bool chol = __bfgs::inv_sympd(H);
            
            __bfgs::bfgs_step<real> step;
            if (chol)
                step.set_hessian(std::move(H));
            else
                step.set_hessian(arma::eye<arma::Mat<real>>(x.size(), x.size()));
            
            return __optim_base::gen_gradient_solve(x, std::forward<Func>(f), std::forward<Grad>(df), std::forward<Hess>(hessian), opts, step);
        }
    } // namespace optimization
} // namespace numerics
#endif

#endif