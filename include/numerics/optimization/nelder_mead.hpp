#ifndef NUMERICS_OPTIMIZATION_NELDER_MEAD_HPP
#define NUMERICS_OPTIMIZATION_NELDER_MEAD_HPP

#include <vector>
#include <random>
#include <algorithm>
#include <set>
#include "numerics/optimization/optim_base.hpp"

namespace numerics {namespace optimization {

template <typename real>
struct NelderMeadOptions : public OptimizationOptions<real>
{
    real step = 1;
    real expand = 2;
    real contract = 0.5;
    real shrink = 0.5;
    real initial_radius = 1;
    int seed = 1;
};

namespace __nelder_mead
{
    template <class vec, typename real>
    struct xy
    {
        vec x;
        real y;
    };

    template<class vec, typename real>
    class comp_xy
    {
    public:
        comp_xy() {}

        bool operator()(const xy<vec,real>& a, const xy<vec,real>& b) const
        {
            return a.y > b.y; // implies reverse ordering
        }
    };

    template <class vec, typename real, std::invocable<vec> Func>
    std::set<xy<vec,real>, comp_xy<vec,real>> init_simplex(const vec& x, Func f, const NelderMeadOptions<real>& opts)
    {
        u_long n = x.size();
        std::default_random_engine gen(opts.seed);
        std::uniform_real_distribution<real> distribution(-1,1);
        auto randu = [&](u_long n) -> vec
        {
            vec v(n);
            for (u_long i=0; i < n; ++i)
                v[i] = distribution(gen);
            return v;
        };

        // generate random vectors and orthogonalized via modified Gram-Schmidt
        std::vector<vec> q(n);
        for (u_long i=0; i < n; ++i)
            q[i] = randu(n);

        for (u_long i=0; i < n; ++i)
        {
            q[i] /= __optim_base::norm_impl(q[i]);
            for (u_long j=i+1; j < n; ++j)
                q[j] -= __optim_base::dot_impl(q[j], q[i]) * q[i];
        }

        // add the random orthogonal directions to initial point x
        std::set<__nelder_mead::xy<vec,real>, __nelder_mead::comp_xy<vec,real>> simplex(comp_xy<vec,real>{});
        for (u_long i=0; i < n; ++i)
        {
            q[i] = x + opts.initial_radius * q[i];
            xy<vec,real> u;
            u.x = std::move(q[i]);
            u.y = f(u.x);
            simplex.insert(std::move(u));
        }

        // append x to the simplex for a total of n+1 points in R^n
        xy<vec,real> u;
        u.x = x;
        u.y = f(x);
        simplex.insert(std::move(u));
        
        return simplex;
    }

    template <class vec, typename real, std::invocable<vec> Func>
    std::set<xy<vec,real>, comp_xy<vec,real>> shrink_simplex(const std::set<xy<vec,real>, comp_xy<vec,real>>& s, Func f, real shrink)
    {
        std::set<xy<vec,real>, comp_xy<vec,real>> s1(comp_xy<vec,real>{});
        
        auto it = s.rbegin();
        xy<vec, real> best = *it;
        ++it;

        for (; it != s.rend(); ++it)
        {
            xy<vec,real> u;
            u.x = best.x + shrink * (it->x - best.x);
            u.y = f(u.x);

            s1.insert(std::move(u));
        }
        
        s1.insert(std::move(best));

        return s1;
    }

    template <class vec, class It>
    vec mean(It it, It last)
    {
        u_long n = 1;
        vec m = it->x;
        ++it;

        for (; it != last; ++it)
            m += (it->x - m)/(++n);

        return m;
    }

} // namespace __nelder_mead

template <class vec, std::invocable<vec> Func, typename real = typename vec::value_type>
OptimizationResults<real,real> nelder_mead(vec& x, Func f, const NelderMeadOptions<real>& opts = {})
{
    if (opts.step <= 0)
        throw std::invalid_argument("nelder_mead() error: require step size (=" + std::to_string(opts.step) + ") > 0");
    if (opts.expand <= 1)
        throw std::invalid_argument("nelder_mead() error: require expansion parameter (=" + std::to_string(opts.expand) + ") > 1");
    if ((opts.contract <= 0) or (opts.contract >= 1))
        throw std::invalid_argument("nelder_mead() error: require 0 < contraction parameter (=" + std::to_string(opts.contract) + ") < 1");
    if ((opts.shrink <= 0) or (opts.shrink >= 1))
        throw std::invalid_argument("nelder_mead() error: require 0 < shrinking parameter (=" + std::to_string(opts.shrink) + ") < 1");
    if (opts.initial_radius <= 0)
        throw std::invalid_argument("nelder_mead() error: require simplex size (=" + std::to_string(opts.initial_radius) + ") > 0");

    u_long m = x.size();

    if (m < 2)
        throw std::invalid_argument("nelder_mead() error: nelder_mead() not suited for one dimensional optimization, use fminsearch() instead.");

    // simplex set is ordered from worst point to best point
    auto simplex = __nelder_mead::init_simplex<vec,real>(x, std::forward<Func>(f), opts);

    real ybest = simplex.rbegin()->y;
    u_long n_iter = 0;
    ExitFlag flag = NONE;

    VerboseTracker T(opts.max_iter);
    if (opts.verbose)
        T.header("f");
    
    while (true)
    {
        auto scndw = ++simplex.begin();
        auto worst = simplex.begin();

        // reflect x(worst) accross center
        vec c = __nelder_mead::mean<vec>(scndw, simplex.end());
        vec xr = c + opts.step * (c - worst->x);
        real yr = f(xr);
        if ((ybest < yr) and (yr < scndw->y))
        {
            __nelder_mead::xy<vec,real> u;
            u.x = std::move(xr);
            u.y = yr;

            simplex.erase(worst);
            simplex.insert(std::move(u));
        }
        else if (yr < ybest)
        { // new point is very good, attempt further search in this direction
            vec xe = c + opts.expand * (xr - c);
            real ye = f(xe);

            __nelder_mead::xy<vec,real> u;

            if (ye < yr)
            {    
                u.x = std::move(xe);
                u.y = ye;
            }
            else
            {
                u.x = std::move(xr);
                u.y = yr;
            }
            
            simplex.erase(worst);
            simplex.insert(std::move(u));
        }
        else if (scndw->y < yr)
        { // potential over shoot
            if (yr < worst->y)
            { // contraction outside simplex
                vec xc = c + opts.contract * (xr - c);
                real yc = f(xc);
                if (yc < yr)
                {
                    __nelder_mead::xy<vec,real> u;
                    u.x = std::move(xc);
                    u.y = yc;

                    simplex.erase(worst);
                    simplex.insert(std::move(u));
                }
                else // shrink simplex
                    simplex = __nelder_mead::shrink_simplex<vec,real>(simplex, std::forward<Func>(f), opts.shrink);
            }
            else
            { // contraction inside simplex
                vec xc = c + opts.contract * (worst->x - c);
                real yc = f(xc);
                if (yc < worst->y)
                {
                    __nelder_mead::xy<vec,real> u;
                    u.x = std::move(xc);
                    u.y = yc;

                    simplex.erase(worst);
                    simplex.insert(std::move(u));
                }
                else // shrink simplex
                    simplex = __nelder_mead::shrink_simplex<vec,real>(simplex, std::forward<Func>(f), opts.shrink);
            }
        }

        ybest = simplex.rbegin()->y;

        if (opts.verbose)
            T.iter(n_iter, ybest);
        ++n_iter;

        real ftol = opts.ftol * std::max<real>(1.0, std::abs(ybest));
        if (std::abs(ybest - simplex.begin()->y) < ftol)
        {
            flag = CONVERGED;
            if (opts.verbose)
                T.success_flag();
            break;
        }

        real xtol = opts.xtol * std::max<real>(1.0, __optim_base::norm_impl(simplex.rbegin()->x));
        if (__optim_base::norm_impl(static_cast<vec>(simplex.rbegin()->x - simplex.begin()->x)) < xtol)
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
    
    x = simplex.rbegin()->x;

    OptimizationResults<real, real> rslts;
    rslts.fval = simplex.rbegin()->y;
    rslts.n_iter = n_iter;
    rslts.flag = flag;

    return rslts;
}

}}

#endif