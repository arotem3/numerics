#ifndef NUMERICS_OPTIMIZATION_BASE
#define NUMERICS_OPTIMIZATION_BASE

#include <cmath>
#include <sstream>
#include <string>
#include <type_traits>
#include <concepts>
#include <functional>

#include "numerics/concepts.hpp"
#include "numerics/vmath.hpp"

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

        template <typename vec>
        struct OptimizationResults
        {
            typedef vec vec_type;

            vec fval;
            u_long n_iter;
            ExitFlag flag;

            // produces a readable message explaining why the solver completed
            // computation.
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

        template <std::floating_point precision>
        struct OptimizationOptions
        {
            typedef precision precision_type;
            
            u_long max_iter = 100;
            precision xtol = std::sqrt(std::numeric_limits<precision>::epsilon());
            precision ftol = std::sqrt(std::numeric_limits<precision>::epsilon());
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
            template <class vec, std::invocable<vec> Func, typename precision = precision_t<typename vec::value_type>>
            inline vec jac_product(const vec& x, const vec& v, const vec& F, Func f)
            {
                precision C = std::sqrt(std::numeric_limits<precision>::epsilon()) * std::max<precision>(1, __vmath::norm_impl(x)) / std::max<precision>(1, __vmath::norm_impl(v));
                return (f(static_cast<vec>(x + C*v)) - F) / C;
            }

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

            template <class vec, typename eT, class Func, class Jac, class Step>
            OptimizationResults<vec> gen_solve(vec& x, Func f, Jac jacobian, const OptimizationOptions<precision_t<eT>>& opts, Step& step)
            {
                typedef precision_t<eT> precision;

                VerboseTracker verbose(opts.max_iter);
                if (opts.verbose)
                    verbose.header("norm(f)");

                vec F = f(x);
                vec dx = eT(0)*x;
                precision f0 = __vmath::norm_impl(F);
                precision ftol = std::max<precision>(1, f0) * opts.ftol;
                precision xtol = std::max<precision>(1, __vmath::norm_impl(x)) * opts.xtol;

                u_long n_iter = 0;
                ExitFlag flag = NONE;
                while (true)
                {
                    bool successful_step = false;

                    try {
                        successful_step = step_impl(step, dx, x, F, std::forward<Func>(f), std::forward<Jac>(jacobian), opts);
                    } catch(const std::exception& e) {
                        std::cerr << "while trying to compute step, an exception was thrown:\n"<< e.what() << '\n';
                    }

                    if (not successful_step)
                    {
                        flag = STEP_FAILED;
                        if (opts.verbose)
                            verbose.failed_step_flag();
                        break;
                    }

                    f0 = __vmath::norm_impl(F);

                    x += dx;
                    ++n_iter;

                    if (opts.verbose)
                        verbose.iter(n_iter, f0);

                    if (f0 < ftol)
                    {
                        flag = CONVERGED;
                        if (opts.verbose)
                            verbose.success_flag();
                        break;
                    }

                    if (__vmath::norm_impl(dx) < xtol)
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

                OptimizationResults<vec> rslts;
                rslts.flag = flag;
                rslts.n_iter = n_iter;
                rslts.fval = std::move(F);
                return rslts;
            }

            template <class vec, std::floating_point eT, class Func, class Grad, class Hess, class Step>
            OptimizationResults<eT> gen_gradient_solve(vec& x, Func f, Grad df, Hess hessian, const OptimizationOptions<eT>& opts, Step& step)
            {
                VerboseTracker T(opts.max_iter);
                if (opts.verbose)
                    T.header("f");
                
                vec dx = eT(0)*x, g = -df(x);

                eT gtol = std::max<eT>(1, __vmath::norm_impl(g)) * opts.ftol;
                eT xtol = std::max<eT>(1, __vmath::norm_impl(x)) * opts.xtol;

                u_long n_iter = 0;
                ExitFlag flag = NONE;
                while (true)
                {
                    bool successful_step = false;

                    try {
                        successful_step = gstep_impl(step, dx, x, g, std::forward<Func>(f), std::forward<Grad>(df), std::forward<Hess>(hessian), opts);
                    } catch(const std::exception& e) {
                        std::cerr << "While trying to compute step, the following exception was thrown:\n" << e.what() << '\n';
                    }

                    if (not successful_step)
                    {
                        flag = STEP_FAILED;
                        if (opts.verbose)
                            T.failed_step_flag();
                        break;
                    }
                    x += dx;
                    ++n_iter;

                    if (opts.verbose)
                        T.iter(n_iter, f(x));

                    if (__vmath::norm_impl(g) < gtol)
                    {
                        flag = CONVERGED;
                        if (opts.verbose)
                            T.success_flag();
                        break;
                    }

                    if (__vmath::norm_impl(dx) < xtol)
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

                OptimizationResults<eT> rslts;
                rslts.fval = f(x);
                rslts.n_iter = n_iter;
                rslts.flag = flag;

                return rslts;
            }

        } // namespace __optim_base
    } // namespace optimization
} // namespace numerics
#endif