#ifndef NUMERICS_OPTIMIZATION_MIXFPI_HPP
#define NUMERICS_OPTIMIZATION_MIXFPI_HPP

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

#include "numerics/concepts.hpp"
#include "numerics/optimization/optim_base.hpp"

#ifdef NUMERICS_WITH_ARMA
namespace numerics {
    namespace optimization {
        template <std::floating_point real>
        struct mixFPI_Options : public OptimizationOptions<real>
        {
            u_long steps = 5;
        };

        // solves x == f(x) using Anderson mixing for accelarated fixed point iteration.
        // The vector x should be intialized to a guess of the solution. Global
        // convergence depends on the quality of this initial guess.
        // see:
        // https://en.wikipedia.org/wiki/Anderson_acceleration
        template <scalar_field_type scalar, std::invocable<arma::Col<scalar>> Func>
        OptimizationResults<arma::Col<scalar>> mixFPI(arma::Col<scalar>& x, Func f, const mixFPI_Options<precision_t<scalar>>& opts={})
        {
            typedef precision_t<scalar> precision;
            u_long n = x.n_elem;

            arma::Mat<scalar> F(n, opts.steps, arma::fill::zeros);
            arma::Mat<scalar> X(n, opts.steps, arma::fill::zeros);
            
            arma::Mat<scalar> FF(n+1, opts.steps, arma::fill::ones);
            
            arma::Col<scalar> b(n+1, arma::fill::zeros);
            b(n) = scalar(1.0f);

            u_long n_iter = 0;
            ExitFlag flag = NONE;
            u_long head = 0;
            VerboseTracker T(opts.max_iter);
            if (opts.verbose)
                T.header("max|x-f(x)|");
            while (true)
            {
                if (opts.verbose)
                    T.iter(n_iter, arma::norm(F.col(head) - x,"inf"));

                head = n_iter % opts.steps;
                
                F.col(head) = f(x);
                X.col(head) = x;

                if (arma::norm(F.col(head) - X.col(head),"inf") < opts.xtol)
                {
                    flag = CONVERGED;
                    if (opts.verbose)
                        T.success_flag();
                    break;
                }

                FF.submat(0,head,n-1,head) = F.col(head) - X.col(head);
                arma::Col<scalar> c;
                bool success;
                if (n_iter < opts.steps)
                    success = arma::solve(c, FF.cols(0,n_iter), b);
                else
                    success = arma::solve(c, FF, b);

                if (not success)
                {
                    flag = STEP_FAILED;
                    T.failed_step_flag();
                    break;
                }

                if (n_iter < opts.steps)
                    x = F.cols(0,n_iter) * c;
                else
                    x = F * c;

                ++n_iter;

                if (n_iter >= opts.max_iter)
                {
                    flag = MAX_ITER;
                    if (opts.verbose)
                        T.max_iter_flag();
                    break;
                }
            }

            OptimizationResults<arma::Col<scalar>> rslts;
            rslts.fval = f(x);
            rslts.flag = flag;
            rslts.n_iter = n_iter;

            return rslts;
        }
    } // namespace optimization
} // namespace numerics
#endif

#endif