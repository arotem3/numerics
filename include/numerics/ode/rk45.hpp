#ifndef NUMERICS_ODE_RK45_HPP
#define NUMERICS_ODE_RK45_HPP

#include <array>
#include <iterator>

#include "numerics/vmath.hpp"
#include "numerics/ode/ode_base.hpp"

namespace numerics
{
    namespace ode
    {
        template <std::floating_point real, typename vec>
        struct rk45Results : public ivpResults<real,vec>
        {
            template <std::floating_point T, typename V, std::invocable<T,V> Func>
            friend rk45Results<T,V> rk45(Func, const std::vector<T>&, const V&, const ivpOpts<T>&, const std::vector<Event<T,V>>&);

        private:
            std::vector<vec> y_half;

        public:
            rk45Results(bool dense=false) : ivpResults<real,vec>(dense) {}

            // C1 interpolation using quartic hermite polynomial
            vec operator()(real s) const override
            {
                if (not this->dense)
                    throw std::runtime_error("rk45Results operator() error: solution is not dense, cannot evaluate.");

                if (s == this->t.front())
                    return this->y.front();
                else if (s == this->t.back())
                    return this->y.back();

                size_t i = std::distance(this->t.begin(), std::lower_bound(this->t.begin(), this->t.end(), s));
                if ((i == 0) or (i == this->t.size()))
                    throw std::runtime_error("rk45Results operator() error: t =" + std::to_string(s) + " outside of solution range (" + std::to_string(this->t.front()) + ", " + std::to_string(this->t.back()) + ").");
                
                --i;
                real k = (this->t.at(i+1) - this->t.at(i));
                real theta = (s - this->t.at(i)) / k;
                // quartic-polynomial hermite interpolation basis
                real b0 = _ivp_helper::hquarticv(theta);
                real b1 = k * _ivp_helper::hquarticd(theta);
                real b2 = _ivp_helper::hquarticc(theta);
                real b3 = -k * _ivp_helper::hquarticd(1-theta);
                real b4 = _ivp_helper::hquarticv(1-theta);

                vec u = b0*this->y.at(i) + b1*this->f.at(i) + b2*y_half.at(i) + b3*this->f.at(i+1) + b4*this->y.at(i+1);
                // vec u = b0*this->y[i] + b1*this->f[i] + b2*y_half[i] + b3*this->f[i+1] + b4*this->y[i+1];
                return u;
            }

            std::vector<vec> operator()(std::vector<real> s) const override
            {
                if (not this->dense)
                    throw std::runtime_error("rk45 operator() error: solution is not dense, cannot evaluate.");
            
                std::sort(s.begin(), s.end());
                std::vector<vec> u;
                u.reserve(s.size());

                if (s.front() < this->t.front() or this->t.back() < s.back()) {
                    std::stringstream err;
                    err << "rk45 operator() error: requested interpolation points t {min(t)=" << s.front() << ", max(t)="
                        << s.back() << "} is out bounds of the computed solution range: (" << this->t.front() << ", " << this->t.back() << ").";
                    throw std::runtime_error(err.str());
                }

                size_t i = std::distance(this->t.begin(), std::lower_bound(this->t.begin(), this->t.end(), s.front()));
                if (i > 0)
                    --i;

                for (real w : s)
                {
                    while (this->t.at(i+1) < w)
                        ++i;
                    
                    real k = this->t.at(i+1) - this->t.at(i);
                    real theta = (w - this->t.at(i)) / k;
                    
                    real b0 = _ivp_helper::hquarticv(theta);
                    real b1 = k*_ivp_helper::hquarticd(theta);
                    real b2 = _ivp_helper::hquarticc(theta);
                    real b3 = -k*_ivp_helper::hquarticd(1-theta);
                    real b4 = _ivp_helper::hquarticv(1-theta);

                    // vec v = b0*this->y[i] + b1*this->f[i] + b2*y_half[i] + b3*this->f[i+1] + b4*this->y[i+1];
                    vec v = b0*this->y.at(i) + b1*this->f.at(i) + b2*y_half.at(i) + b3*this->f.at(i+1) + b4*this->y.at(i+1);
                    u.push_back(std::move(v));
                }

                return u;
            }
        };

        // Dormand Prince adaptive Runge-Kutta fourth order method.
        // see pg 178 of:
        // "Solving Ordinary Differential Equations I" by E. Hairer, and G.
        // Wanner
        template <std::floating_point real, typename vec, std::invocable<real,vec> Func>
        rk45Results<real,vec> rk45(Func f, const std::vector<real>& tspan, const vec& y0, const ivpOpts<real>& opts = {}, const std::vector<Event<real,vec>>& events = {})
        {
            constexpr real rka[5][5] = {
                {           0.2,             0.0,            0.0,          0.0,             0.0},
                {      3.0/40.0,        9.0/40.0,            0.0,          0.0,             0.0},
                {     44.0/45.0,      -56.0/15.0,       32.0/9.0,          0.0,             0.0},
                {19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0,             0.0},
                { 9017.0/3168.0,     -355.0/33.0, 46732.0/5247.0,   49.0/176.0, -5103.0/18656.0}
            };
            constexpr real rk5b[] = {35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0};
            constexpr real rk4b[] = {5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0};
            constexpr real rkc[] = {0.2, 0.3, 0.8, 8.0/9.0, 1.0};
            constexpr real denseb[] = {
                0.5*rk5b[0] + 0.05456140216335728329l,
                0.0,
                0.5*rk5b[2] + 0.16721403027197487165l,
                0.5*rk5b[3] - 0.35534543509928150115l,
                0.5*rk5b[4] + 0.22012077299504946102l,
                0.5*rk5b[5] - 0.11045507856723408743l,
                0.023904308236133972612l
            };

            _ivp_helper::check_range<real>(tspan);

            auto T = tspan.begin();

            _ivp_helper::khan_stable_sum<real> _add; // helper variable for numerically accumulating t
            real t = *T;
            vec y = y0;
            vec F = f(t, y);
            
            real k = (opts.initial_step <= 0) ? real(0.01)*(tspan[1] - tspan[0]) : opts.initial_step;
            const real max_step = (opts.max_step <= 0) ? real(0.1)*(tspan.back() - tspan.front()) : opts.max_step;

            rk45Results<real,vec> sol(opts.dense_output);
            sol.t.push_back(t);
            sol.y.push_back(y);
            sol.f.push_back(F);

            ++T;
            for (; T != tspan.end(); ++T)
            {
                real tf = *T;
                while (t < tf)
                {
                    real tol = std::max<real>(opts.atol, opts.rtol*__vmath::norm_impl(y));
                        
                    while (true) // take step and see if it is acceptable
                    {
                        if (std::abs(tf - (t + k)) < (std::abs(t)+1)*std::numeric_limits<real>::epsilon())
                            k = tf - t;
                        k = std::min(k, tf - t);

                        vec z[7];
                        z[0] = k * F;

                        #pragma unroll
                        for (int i=0; i < 5; ++i)
                        {
                            vec p = y;
                            #pragma unroll
                            for (int j=0; j <= i; ++j)
                                p += rka[i][j]*z[j];

                            real s = t + rkc[i]*k;

                            z[i+1] = k * f(s, p);
                        }

                        vec rk5 = y;
                        #pragma unroll
                        for (int i : {0,2,3,4,5}) // skip 1 because rk4b[1] == 0
                            rk5 += rk5b[i]*z[i];
                        z[6] = k * f(real(t+k), rk5);

                        vec rk4 = y;
                        #pragma unroll
                        for (int i : {0,2,3,4,5,6}) // skip 1 because rk5b[1] == 0
                            rk4 += rk4b[i]*z[i];

                        real res = __vmath::norm_impl(static_cast<vec>(rk5-rk4));
                        real k1 = k * std::min<real>(10.0, std::max<real>(0.1, real(0.9)*std::pow(tol/res, real(0.2))));
                        k1 = std::min<real>(k1, max_step);

                        if (k1 < (std::abs(t)+1)*std::numeric_limits<real>::epsilon()) {
                            sol.flag = ExitFlag::STEP_FAIL;
                            return sol;
                        }

                        if (res > tol) {
                            k = k1;
                            continue;
                        }
                        
                        real alpha = 1.;
                        auto event_out = _ivp_helper::handle_event(t, y, t+k, rk5, events, opts.event_tol);
                        
                        if ((0 < alpha) and (alpha < 1)) // close to event, need to refine step
                            k *= alpha;
                        else { // advance solution
                            if (opts.dense_output) {
                                vec yhalf = y;
                                #pragma unroll
                                for (int i : {0,2,3,4,5,6}) // skip 1, because denseb[1] == 0
                                    yhalf += denseb[i]*z[i];
                                sol.y_half.push_back(std::move(yhalf));
                            }

                            t = _add(t, k);
                            y = std::move(rk5);
                            F = f(t, y);
                            k = k1;

                            bool is_grid = std::abs(tf - t) < std::abs(tf+1)*std::numeric_limits<real>::epsilon();
                            if (sol.stopping_event or is_grid or !opts.grid_only or opts.dense_output) {
                                sol.t.push_back(t);
                                sol.y.push_back(y);
                                sol.f.push_back(F);

                                if (sol.stopping_event)
                                    return sol;
                            }

                            break;
                        }
                    }
                }
            }

            sol.flag = ExitFlag::SUCCESS;
            return sol;
        }
    } // namespace ode
    
} // namespace numerics


#endif