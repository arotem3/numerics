#ifndef NUMERICS_ODE_BASE_HPP
#define NUMERICS_ODE_BASE_HPP

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

#include <concepts>
#include <cmath>
#include <vector>
#include <limits>
#include <stdexcept>
#include <optional>
#include <functional>
#include <utility>
#include <complex>
#include <sstream>

#include "numerics/concepts.hpp"

namespace numerics {
    namespace ode {
        template <std::floating_point real>
        struct ivpOpts
        {
            real atol = std::sqrt(std::numeric_limits<real>::epsilon());
            real rtol = 1e-4f;
            u_long solver_max_iter = 25; // for implicit methods only
            bool solver_quasi_newton = true; // for implicit methods with armadillo-types only: if quasi_newton solver, then Broyden's method will be used. Should be more efficient per-iteration
            real event_tol = std::sqrt(std::numeric_limits<real>::epsilon());
            bool dense_output = false;
            bool grid_only = false;
            real initial_step = 0;
            real max_step = 0;
        };



        enum struct EventDir
        {
            ALL = 0,
            INC = 1,
            DEC = -1
        };

        template <std::floating_point real, typename vec>
        struct Event
        {
            EventDir dir;
            std::function<real(real, const vec&)> event;
        };

        enum struct ExitFlag
        {
            NONE,
            SUCCESS,
            STEP_FAIL,
            NL_FAIL,
            EVENT
        };

        template <std::floating_point real, typename vec>
        struct ivpResults
        {
            std::vector<real> t;
            std::vector<vec>  y; // y(t)
            std::vector<vec>  f; // f(t,y(t))
            std::optional<u_long> stopping_event;
            ExitFlag flag;
            const bool dense;

            ivpResults(bool dense=false) : dense(dense), flag(ExitFlag::NONE) {}

            std::string get_flag() const
            {
                switch (flag)
                {
                    case ExitFlag::NONE:
                        return "solver never called?";
                        break;
                    case ExitFlag::SUCCESS:
                        return "solver computed solution successfully.";
                        break;
                    case ExitFlag::STEP_FAIL:
                        return "solver failed because step-size less than machine epsilon.";
                        break;
                    case ExitFlag::NL_FAIL:
                        return "solver failed because non-linear solver could not produce a solution.";
                        break;
                    case ExitFlag::EVENT:
                        return "solver encountered event.";
                        break;
                }
                return "";
            }

            // C1 interpolation using cubic Hermite spline
            virtual vec operator()(real s) const
            {
                if (not dense)
                    throw std::runtime_error("ivpResults operator() error: solution is not dense, cannot evaluate.");

                if (s == t.front())
                    return y.front();
                else if (s == t.back())
                    return y.back();

                size_t i = std::distance(t.begin(), std::lower_bound(t.begin(), t.end(), s));
                if ((i == 0) or (i == t.size()))
                    throw std::runtime_error("ivpResults operator() error: t =" + std::to_string(s) + " outside of solution range (" + std::to_string(t.front()) + ", " + std::to_string(t.back()) + ").");
                
                --i;
                real theta = (s - t.at(i)) / (t.at(i+1) - t.at(i));
                real h = t.at(i+1) - t.at(i);
                real b0 = (1+2*theta)*(1-theta)*(1-theta);
                real b1 = theta*(1-theta)*(1-theta);
                real b2 = theta*theta*(3-2*theta);
                real b3 = theta*theta*(theta-1);

                vec u = b0*y.at(i) + h*b1*f.at(i) + b2*y.at(i+1) + h*b3*f.at(i+1);
                return u;
            }

            virtual std::vector<vec> operator()(std::vector<real> s) const
            {
                if (not dense)
                    throw std::runtime_error("ivpResults operator() error: solution is not dense, cannot evaluate.");
                
                std::sort(s.begin(), s.end());    
                std::vector<vec> u;
                u.reserve(s.size());

                if (s.front() < t.front() or t.back() < s.back()) {
                    std::stringstream err;
                    err << "ivpResults operator() error: requested interpolation points t {min(t)=" << t.front() << ", max(t)="
                        << t.back() << "} is out bounds of the computed solution range: (" << t.front() << ", " << t.back() << ").";
                    throw std::runtime_error(err.str());
                }

                size_t i = std::distance(t.begin(), std::lower_bound(t.begin(), t.end(), s.front()));
                if (i > 0)
                    --i;

                auto phi = [](real w) -> real {return (1 + 2*w)*(1-w)*(1-w);}; // phi(0) = 1, phi'(0) = 0, phi(1) = 0, phi'(1) = 0
                auto psi = [](real w) -> real {return w*(1-w)*(1-w);}; // psi(0) = 0, psi'(0) = 1, psi(1) = 0, psi'(1) = 0
                
                for (real w : s)
                {
                    while (t.at(i+1) < w)
                        ++i;

                    real theta = (w - t.at(i)) / (t.at(i+1) - t.at(i));
                    real h = t.at(i+1) - t.at(i);
                    // real theta = (w - t[i]) / (t[i+1] - t[i]);
                    vec v = phi(theta)*y.at(i) + h*psi(theta)*f.at(i) + phi(1-theta)*y.at(i+1) - h*psi(1-theta)*f.at(i+1);
                    // vec v = phi(theta)*y[i] + h*psi(theta)*f[i] + phi(1-theta)*y[i+1] - h*psi(1-theta)*f[i+1];
                    u.push_back(std::move(v));
                }

                return u;
            }
        };

        namespace _ivp_helper
        {
            template <std::floating_point real>
            void check_range(const std::vector<real>& tspan)
            {
                if (tspan.size() < 2)
                    throw std::runtime_error("initial-value-problem error: tspan must include at least a starting value and an ending value");
                if (not std::is_sorted(tspan.begin(), tspan.end()))
                    throw std::runtime_error("initial-value-problem error: tspan must be sorted");
            }

            template <std::floating_point real, typename vec>
            std::optional<std::pair<u_long,real>> handle_event(real t0, const vec& y0, real t1, const vec& y1, const std::vector<Event<real, vec>>& events, real tol)
            {
                u_long e;
                real a = std::numeric_limits<real>::infinity();

                for (u_long i=0; i < events.size(); ++i)
                {
                    real g0 = events[i].event(t0, y0);
                    real g1 = events[i].event(t1, y1);

                    if ((g0 < 0) != (g1 < 0)) { // event encountered
                        if (std::abs(g1) <= tol) { // stopping criteria possible
                            bool all = (events[i].dir == EventDir::ALL);
                            bool dec = (g1 < 0) and (events[i].dir == EventDir::DEC);
                            bool inc = (g1 > 0) and (events[i].dir == EventDir::INC);
                            if (all or dec or inc)
                                return std::make_optional(std::make_pair(i, real(0.0)));
                        } else {
                            real alpha = std::min<real>(1, real(0.95)*std::abs(g1) / (std::abs(g1) + std::abs(g0)));
                            if (alpha < a) {
                                a = alpha;
                                e = i;
                            }
                        }
                    }
                }

                if (a < std::numeric_limits<real>::infinity())
                    return std::make_optional(std::make_pair(e, a));
                else
                    return std::nullopt;
            }

            // we want to accurately accumulate t in the integrators, so we use a stable sum
            // https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            template <std::floating_point real>
            class khan_stable_sum
            {
            private:
                real c = 0;

            public:
                inline real operator()(real s, real k)
                {
                    real y = k - c;
                    real t = s + y;
                    c = (t - s) - y;
                    return t;
                }
            };

            // hermite quartic polynomial which collocates a function at
            // {0,1/2,1} and its derivative at {0,1}. This function evaluates
            // the basis function for the Value at zero, for the value at 1 use
            // hquarticv(1-w).
            template <std::floating_point real>
            inline real hquarticv(real w)
            {
                return -2*(4*w+1)*(w-1)*(w-1)*(w-real(0.5));
            }

            // hermite quartic polynomial which collocates a function at
            // {0,1/2,1} and its derivative at {0,1}. This function evaluates
            // the basis function for the Derivative at zero, for the derivative
            // at 1 use -hquarticd(1-w).
            template <std::floating_point real>
            inline real hquarticd(real w)
            {
                return -2*w*(w-1)*(w-1)*(w-real(0.5));
            }

            // hermite quartic polynomial which collocates a function at
            // {0,1/2,1} and its derivative at {0,1}. This function evaluates
            // the basis function for the value at 1/2 (the Center).
            template <std::floating_point real>
            inline real hquarticc(real w)
            {
                return 16*w*w*(w-1)*(w-1);
            }
        } // namespace __ivp_helper
    } // namespace ode
} // namespace numerics

#endif