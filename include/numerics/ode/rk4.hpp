#ifndef NUMERICS_ODE_RK4_HPP
#define NUMERICS_ODE_RK4_HPP

#include "numerics/ode/ode_base.hpp"

namespace numerics
{
    namespace ode
    {
        // single step of the five stage fourth order low storage Runge Kutta
        // method.
        // see:
        //"Fourth-order 2N-storage Runge-Kutta schemes" M.H. Carpenter and C.
        // Kennedy
        template<std::floating_point real, typename vec, std::invocable<real,vec> Func>
        inline void rk4_step(Func f, real t, vec& y, real k, const vec& F0)
        {
            static const real rk4a[5] = {0.0, -567301805773.0/1357537059087.0, -2404267990393.0/2016746695238.0, -3550918686646.0/2091501179385.0, -1275806237668.0/842570457699.0};
            static const real rk4b[5] = {1432997174477.0/9575080441755.0, 5161836677717.0/13612068292357.0, 1720146321549.0/2090206949498.0, 3134564353537.0/4481467310338.0, 2277821191437.0/14882151754819.0};
            static const real rk4c[5] = {0.0, 1432997174477.0/9575080441755.0, 2526269341429.0/6820363962896.0, 2006345519317.0/3224310063776.0, 2802321613138.0/2924317926251.0};

            vec v = k * F0;
            y += rk4b[0]*v;
            #pragma unroll
            for (int i=1; i < 5; ++i)
            {
                real s = t + rk4c[i]*k;
                v = rk4a[i]*v + k*f(s, y);
                y += rk4b[i]*v;
            }
        }

        // Five stage fourth order low storage rk4,
        // see:
        //"Fourth-order 2N-storage Runge-Kutta schemes" M.H. Carpenter and C.
        // Kennedy
        template <std::floating_point real, typename vec, std::invocable<real,vec> Func>
        ivpResults<real,vec> rk4(Func f, const std::vector<real>& tspan, const vec& y0, real k, const ivpOpts<real>& opts = {}, const std::vector<Event<real,vec>>& events = {})
        {
            static const real rk4a[5] = {0.0, -567301805773.0/1357537059087.0, -2404267990393.0/2016746695238.0, -3550918686646.0/2091501179385.0, -1275806237668.0/842570457699.0};
            static const real rk4b[5] = {1432997174477.0/9575080441755.0, 5161836677717.0/13612068292357.0, 1720146321549.0/2090206949498.0, 3134564353537.0/4481467310338.0, 2277821191437.0/14882151754819.0};
            static const real rk4c[5] = {0.0, 1432997174477.0/9575080441755.0, 2526269341429.0/6820363962896.0, 2006345519317.0/3224310063776.0, 2802321613138.0/2924317926251.0};

            _ivp_helper::check_range<real>(tspan);

            auto T = tspan.begin();

            _ivp_helper::khan_stable_sum<real> _add; // helper variable for numerically accumulating t
            real t = *T;
            const real dt = k;
            vec y = y0;
            vec F = f(t, y);

            ivpResults<real,vec> sol(opts.dense_output);
            sol.t.push_back(t);
            sol.y.push_back(y);
            sol.f.push_back(F);

            ++T;
            for (; T != tspan.end(); ++T)
            {
                real tf = *T;
                k = dt;
                while (t < tf)
                {
                    while (true) // take step and see if it is acceptable
                    {
                        if (std::abs(tf - (t + k)) < (std::abs(tf)+1)*std::numeric_limits<real>::epsilon())
                            k = tf - t;
                        k = std::min(k, tf - t);

                        vec v = k * F;
                        vec p = y + rk4b[0]*v;
                        #pragma unroll
                        for (int i=1; i < 5; ++i)
                        {
                            real s = t + rk4c[i]*k;
                            v = rk4a[i]*v + k*f(s, p);
                            p += rk4b[i]*v;
                        }

                        real alpha = 1.;
                        auto event_out = _ivp_helper::handle_event(t, y, t+k, p, events, opts.event_tol);
                        if (event_out) {
                            alpha = event_out->second;
                            if (alpha == 0)
                                sol.stopping_event = event_out->first;
                        }

                        if ((0 < alpha) and (alpha < 1))
                            k *= alpha;
                        else {
                            t = _add(t, k);
                            y = std::move(p);
                            F = f(t, y);

                            bool is_grid = std::abs(tf - t) < std::abs(tf+1)*std::numeric_limits<real>::epsilon();
                            if (sol.stopping_event or is_grid or !opts.grid_only) {
                                sol.t.push_back(t);
                                sol.y.push_back(y);
                                sol.f.push_back(F);

                                if (sol.stopping_event)
                                    return sol;
                            }

                            break;
                        }
                    } // single step
                } // t < tf
            } // T

            sol.flag = ExitFlag::SUCCESS;
            return sol;
        }
    } // namespace ode
} // namespace numerics

#endif