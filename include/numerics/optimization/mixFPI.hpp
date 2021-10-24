#ifndef NUMERICS_OPTIMIZATION_MIXFPI_HPP
#define NUMERICS_OPTIMIZATION_MIXFPI_HPP

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

#include "numerics/optimization/optim_base.hpp"

#ifdef NUMERICS_WITH_ARMA
namespace numerics {
namespace optimization
{

/* Anderson mixing fixed point iteration. Finds solutions of the problem x = f(x).
 * --- x : initial guess and solution output.
 * --- f : vector function of x = f(x). */
template <int steps, typename real, class Func>
OptimizationResults<arma::Col<real>, real> mixFPI(arma::Col<real>& x, Func f, const OptimizationOptions<real>& opts=OptimizationOptions<real>())
{
    u_long n = x.n_elem;

    arma::Mat<real> F = arma::zeros<arma::Mat<real>>(n, steps);
    arma::Mat<real> X = arma::zeros<arma::Mat<real>>(n, steps);
    
    arma::Mat<real> FF = arma::ones<arma::Mat<real>>(n+1, steps);
    
    arma::Col<real> b = arma::zeros<arma::Mat<real>>(n+1);
    b(n) = 1;

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

        head = n_iter % steps;
        
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
        arma::Col<real> c;
        bool success;
        if (n_iter < steps)
            success = arma::solve(c, FF.cols(0,n_iter), b);
        else
            success = arma::solve(c, FF, b);

        if (not success)
        {
            flag = STEP_FAILED;
            T.failed_step_flag();
            break;
        }

        if (n_iter < steps)
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

    OptimizationResults<arma::Col<real>, real> rslts;
    rslts.fval = f(x);
    rslts.flag = flag;
    rslts.n_iter = n_iter;

    return rslts;
}

}
}
#endif

#endif