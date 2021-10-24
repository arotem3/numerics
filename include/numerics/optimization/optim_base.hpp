#ifndef NUMERICS_OPTIMIZATION_BASE
#define NUMERICS_OPTIMIZATION_BASE

#include <cmath>
#include <sstream>
#include <string>
#include <type_traits>
#include <functional>

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

#include "numerics/optimization/VerboseTracker.hpp"

namespace numerics {
namespace optimization {

enum ExitFlag
{
    CONVERGED,
    MIN_STEP_SIZE,
    MAX_ITER,
    STEP_FAILED,
    NONE
};

template <typename vec, typename real=typename vec::value_type>
struct OptimizationResults
{
    typedef vec vec_type;
    typedef real value_type;

    vec fval;
    u_long n_iter;
    ExitFlag flag;

    std::string get_exit_flag() const
    {
        std::stringstream msg;
        msg << "after " << n_iter << " iterations, ";
        if (flag == CONVERGED)
            msg << "first order conditions satisfied within ftol.";
        else if (flag == MIN_STEP_SIZE)
            msg << "solution could not be improved (step size < xtol).";
        else if (flag == MAX_ITER)
            msg << "maximum number of iterations reached.";
        else if (flag == STEP_FAILED)
            msg << "failed to compute step.";
        else {
            msg.str("solver never called.");
        }
        return msg.str();
    }
};

template <typename real>
struct OptimizationOptions
{
    typedef real value_type;
    
    u_long max_iter = 100;
    real xtol = 100*std::sqrt(std::numeric_limits<real>::epsilon());
    real ftol = 100*std::sqrt(std::numeric_limits<real>::epsilon());
    bool verbose = false;

    template <typename T>
    operator OptimizationOptions<T>() const
    {
        OptimizationOptions<T> opts;
        opts.ftol = ftol;
        opts.xtol = xtol;
        opts.max_iter = max_iter;
        opts.verbose = verbose;
        return opts;
    }
};

namespace __optim_base 
{    
    #ifdef NUMERICS_WITH_ARMA
    template <typename real>
    real dot_impl(const arma::Mat<real>& x, const arma::Mat<real>& y)
    {
        return arma::dot(x, y);
    }

    template <typename real>
    real norm_impl(const arma::Mat<real>& x)
    {
        return arma::norm(x);
    }
    #endif

    template <class real, typename = typename std::enable_if<std::is_arithmetic<real>::value, real>::type>
    real dot_impl(const real& x, const real& y)
    {
        return x*y;
    }

    template <class vec, typename real=typename vec::value_type>
    real dot_impl(const vec& x, const vec& y)
    {
        real dot = 0;
        for (u_long i=0; i < x.size(); ++i)
            dot += x[i] * y[i];
        return dot;
    }

    template <class vec, typename real=typename vec::value_type>
    real norm_impl(const vec& x)
    {
        return std::sqrt(dot_impl(x,x));
    }

    #ifdef NUMERICS_WITH_ARMA
    template <typename real>
    inline bool solve_impl(arma::Col<real>& x, const arma::Mat<real>& A, const arma::Col<real>& b)
    {
        return arma::solve(x, A, b);
    }

    template <typename real>
    inline bool solve_impl(arma::Col<real>& x, const arma::SpMat<real>& A, const arma::Col<real>& b)
    {
        return arma::spsolve(x, A, b);
    }

    #endif

    template <class Step, class vec, typename real, class Func, class Jac>
    inline bool step_impl(Step& step, vec& dx, vec& x, vec& F, Func f, Jac jacobian, const OptimizationOptions<real>& opts)
    {
        return step(dx, x, F, std::forward<Func>(f), std::forward<Jac>(jacobian), opts);
    }

    template <class Step, class vec, typename real, class Func>
    inline bool step_impl(Step& step, vec& dx, vec& x, vec& F, Func f, int jacobian, const OptimizationOptions<real>& opts)
    {
        return step(dx, x, F, std::forward<Func>(f), opts);
    }

    template <class Step, class vec, typename real, class Func, class Grad, class Hess>
    inline bool gstep_impl(Step& step, vec& dx, vec& x, vec& g, Func f, Grad df, Hess hessian, const OptimizationOptions<real>& opts)
    {
        return step(dx, x, g, std::forward<Func>(f), std::forward<Grad>(df), std::forward<Hess>(hessian), opts);
    }

    template <class Step, class vec, typename real, class Func, class Grad>
    inline bool gstep_impl(Step& step, vec& dx, vec& x, vec& g, Func f, Grad df, int hessian, const OptimizationOptions<real>& opts)
    {
        return step(dx, x, g, std::forward<Func>(f), std::forward<Grad>(df), opts);
    }

    template <class vec, typename real, class Func, class Jac, class Step>
    OptimizationResults<vec, real> gen_solve(vec& x, Func f, Jac jacobian, const OptimizationOptions<real>& opts, Step& step)
    {
        VerboseTracker verbose(opts.max_iter);
        if (opts.verbose)
            verbose.header("max|f|");

        vec F = f(x);
        vec dx = 0*x;
        real f0, f1 = norm_impl(F);
        real xtol = opts.xtol * std::max<real>(norm_impl(x), 1.0f);

        u_long n_iter = 0;
        ExitFlag flag = NONE;
        while (true)
        {
            f0 = f1;

            bool successful_step = step_impl(step, dx, x, F, std::forward<Func>(f), std::forward<Jac>(jacobian), opts);

            if (not successful_step)
            {
                flag = STEP_FAILED;
                if (opts.verbose)
                    verbose.failed_step_flag();
                break;
            }

            real f1 = norm_impl(F);
            real df = std::abs(f1 - f0);

            x += dx;
            ++n_iter;

            if (opts.verbose)
                verbose.iter(n_iter, f1);

            if (df < opts.ftol)
            {
                flag = CONVERGED;
                if (opts.verbose)
                    verbose.success_flag();
                break;
            }

            if (norm_impl(dx) < xtol)
            {
                flag = MIN_STEP_SIZE;
                if (opts.verbose)
                    verbose.min_step_flag();
                break;
            }

            if (n_iter >= opts.max_iter)
            {
                flag = MAX_ITER;
                if (opts.verbose)
                    verbose.max_iter_flag();
                break;
            }
        }

        OptimizationResults<vec, real> rslts;
        rslts.flag = flag;
        rslts.n_iter = n_iter;
        rslts.fval = std::move(F);
        return rslts;
    }

    template <class vec, typename real, class Func, class Grad, class Hess, class Step>
    OptimizationResults<real,real> gen_gradient_solve(vec& x, Func f, Grad df, Hess hessian, const OptimizationOptions<real>& opts, Step& step)
    {
        VerboseTracker T(opts.max_iter);
        if (opts.verbose)
            T.header("f");
        
        vec dx = 0*x, g;

        u_long n_iter = 0;
        ExitFlag flag = NONE;
        while (true)
        {
            bool successful_step = gstep_impl(step, dx, x, g, std::forward<Func>(f), std::forward<Grad>(df), std::forward<Hess>(hessian), opts);

            if (not successful_step)
            {
                flag = STEP_FAILED;
                if (opts.verbose)
                    T.failed_step_flag();
                break;
            }
            real xtol = opts.xtol*std::max<real>(1.0, norm_impl(x));
            x += dx;
            ++n_iter;

            if (opts.verbose)
                T.iter(n_iter, f(x));

            if (norm_impl(g) < opts.ftol)
            {
                flag = CONVERGED;
                if (opts.verbose)
                    T.success_flag();
                break;
            }

            if (norm_impl(dx) < xtol)
            {
                flag = MIN_STEP_SIZE;
                if (opts.verbose)
                    T.min_step_flag();
                break;
            }

            if (n_iter >= opts.max_iter)
            {
                flag = MAX_ITER;
                if (opts.verbose)
                    T.max_iter_flag();
                break;
            }
        }

        OptimizationResults<real,real> rslts;
        rslts.fval = f(x);
        rslts.n_iter = n_iter;
        rslts.flag = flag;

        return rslts;
    }

}

}
}
#endif