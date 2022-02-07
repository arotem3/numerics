#ifndef NUMERICS_ODE_TRBDF2_HPP
#define NUMERICS_ODE_TRBDF2_HPP

#include <armadillo>
#define NUMERICS_WITH_ARMA

#include "numerics/ode/ode_base.hpp"
#include "numerics/optimization/fzero.hpp"
#include "numerics/optimization/newton.hpp"
#include "numerics/derivatives.hpp"

namespace numerics
{
    namespace ode
    {
        template <typename vec>
        class __rk2_step
        {
        public:
            template <std::floating_point real, std::invocable<real,vec> Func, typename Dummy>
            __rk2_step(real t, const vec& y, Func f, Dummy jac_dummy) {}

            template <std::floating_point real, std::invocable<real,vec> Func>
            std::tuple<bool,real,vec,vec> operator()(real t, const vec& y, const vec& F, real k, Func f, int jac_dummy, const ivpOpts<real>& opts)
            {
                constexpr real C = 2.0 - sqrt(2.0);
                vec y1 = y + real(k*C/2)*F;
                vec Y, F1;
                real s = t + k;

                auto res = [&](vec z) -> vec
                {
                    Y = y1 + real(C/2)*z;
                    F1 = f(s, Y);
                    return z - k*F1;
                };

                vec z2 = real(0)*y;

                optimization::OptimizationOptions<real> optim_opts;
                optim_opts.ftol = opts.rtol / 5;
                optim_opts.max_iter = opts.solver_max_iter;
                optimization::OptimizationResults<vec> rslts = optimization::newton(z2, res, optim_opts);

                if (rslts.flag != optimization::ExitFlag::CONVERGED)
                    return std::make_tuple(false, real(0), vec{}, vec{});

                constexpr real a = (1-C)/(2*C);
                y1 = y + real(a*k)*F + a*z2;
                s = t + k;
                F1 = f(s, y1);

                vec z3 = real(0)*y;
                rslts = optimization::newton(z3, res, optim_opts);

                if (rslts.flag != optimization::ExitFlag::CONVERGED)
                    return std::make_tuple(false, real(0), vec{}, vec{});

                vec err = real(k*(C-1)/3)*F + real(1.0/3.0)*z2 + real(-C/3)*z3;

                real abs_err = __vmath::norm_impl(err);
                return std::make_tuple(true, abs_err, std::move(Y), std::move(F1));
            }
        };

        template <scalar_field_type T>
        class __rk2_step<T>
        {
        public:
            template <std::floating_point real, std::invocable<real,T> Func, typename Dummy>
            __rk2_step(real t, const T& y, Func f, Dummy jac_dummy) {}

            template <std::floating_point real, std::invocable<real,T> Func>
            std::tuple<bool,real,T,T> operator()(real t, const T& y, const T& F, real k, Func f, int jac_dummy, const ivpOpts<real>& opts)
            {
                constexpr real C = 2.0 - sqrt(2.0);
                const real tol = std::max<real>(std::abs(y) * opts.rtol, opts.atol);
                T z1 = k*F;
                T y1 = y + T(C/2)*z1;
                T Y, F1;
                real s = t + k;

                auto res = [&](T z) -> T
                {
                    Y = y1 + T(C/2)*z;
                    F1 = f(s,Y); 
                    return z - k*F1;
                };

                T z2 = optimization::newton_1d(res, T(0), tol/5, opts.solver_max_iter);
                if (std::abs(res(z2)) > tol/5)
                    return std::make_tuple(false, real(0), T{}, T{});

                constexpr real a = (1-C)/(2*C);
                y1 = y + a*z1 + a*z2;
                s = t + k;
                F1 = f(s, y1);

                T z3 = optimization::newton_1d(res, T(0), tol/5, opts.solver_max_iter);
                if (std::abs(res(z3)) > tol/5)
                    return std::make_tuple(false, real(0), T{}, T{});

                T err = T(k*(C-1)/3)*F + T(1.0/3.0)*z2 + T(-C/3)*z3;

                real abs_err = std::abs(err);
                
                return std::make_tuple(true, abs_err, Y, F1);
            }

            template <std::floating_point real, std::invocable<real,T> Func, std::invocable<real,T> Jacobian>
            std::tuple<bool,real,T,T> operator()(real t, const T& y, const T& F, real k, Func f, Jacobian jacobian, const ivpOpts<real>& opts)
            {
                constexpr real C = 2.0 - sqrt(2.0);
                const real tol = std::max<real>(std::abs(y) * opts.rtol, opts.atol);
                T z1 = k*F;
                T y1 = y + T(C/2)*z1;
                T Y, F1;
                T s = t + k;

                auto res = [&](T z) -> T
                {
                    Y = y1 + T(C/2)*z;
                    F1 = f(s,Y); 
                    return z - k*F1;
                };

                auto d_res = [&](T z) -> T
                {
                    return T(1.) - T(k*C/2) * jacobian(s, Y);
                };

                T z2 = optimization::newton_1d(res, d_res, T(0), tol/5, opts.solver_max_iter);
                if (std::abs(res(z2)) > tol/5)
                    return std::make_tuple(false, real(0), T{}, T{});

                constexpr real a = (1-C)/(2*C);
                y1 = y + a*z1 + a*z2;
                s = t + k;
                F1 = f(s, y1);

                T z3 = optimization::newton_1d(res, d_res, T(0), tol/5, opts.solver_max_iter);
                if (std::abs(res(z3)) > tol/5)
                    return std::make_tuple(false, real(0), T{}, T{});

                T err = T(k*(C-1)/3)*F + T(1.0/3.0)*z2 + T(-C/3)*z3;
                err /= T(1.) - T(k*C/2) * jacobian(s, Y);

                real abs_err = std::abs(err);
                
                return std::make_tuple(true, abs_err, Y, F1);
            }
        };

        template <scalar_field_type eT>
        class __rk2_step<arma::Col<eT>>
        {
        private:
            arma::Mat<eT> J;

        public:
            template <std::floating_point real, std::invocable<real,arma::Col<eT>> Func, std::invocable<real,arma::Col<eT>> Jacobian>
            __rk2_step(real t, const arma::Col<eT>& y, Func f, Jacobian jacobian)
            {
                J = jacobian(t, y);
            }

            template <std::floating_point real, std::invocable<real,arma::Col<eT>> Func>
            __rk2_step(real t, const arma::Col<eT>& y, Func f, int jac_dummy)
            {
                auto F = [&](const arma::Col<eT>& u) -> arma::Col<eT>
                {
                    return f(t, u);
                };

                J = jacobian(F, y);
            }

            template <std::floating_point real, std::invocable<real,arma::Col<eT>> Func, typename Jacobian>
            std::tuple<bool,real,arma::Col<eT>,arma::Col<eT>> operator()(real t, const arma::Col<eT>& y, const arma::Col<eT>& F, real k, Func f, Jacobian jacobian, const ivpOpts<real>& opts)
            {
                constexpr real C = 2.0 - sqrt(2.0);
                const real tol = std::max<real>(arma::norm(y) * opts.rtol, opts.atol);

                arma::Mat<eT> L,U,P;
                arma::Mat<eT> A = eT(-k*C/2)*J;
                A.diag() += eT(1.0); // I - k*t*J
                bool success = arma::lu(L, U, P, A);
                if (not success)
                    return std::make_tuple(false, real(0), arma::Col<eT>{}, arma::Col<eT>{});

                // solve TR step
                arma::Col<eT> y1 = y + eT(k*C/2)*F;
                real T = t + C*k;
                arma::Col<eT> F1 = f(T, y1);
                arma::Col<eT> z2(y.n_elem, arma::fill::zeros);
                arma::Col<eT> Y;

                real res = std::numeric_limits<real>::infinity();
                
                for (u_long it=0; it < opts.solver_max_iter; ++it)
                {
                    arma::Col<eT> r = P*(z2 - k * F1);
                    r = arma::solve(arma::trimatl(L), r);
                    r = arma::solve(arma::trimatu(U), r);

                    res = arma::norm(r);

                    z2 -= r;

                    if (res < tol/5)
                        break;
                    else {
                        Y = y1 + eT(C/2)*z2;
                        F1 = f(T, Y);
                    }
                }

                if (res > tol/5)
                    return std::make_tuple(false, real(0), arma::Col<eT>{}, arma::Col<eT>{});

                // solve BDF step
                constexpr real a = (1-C)/(2*C);
                y1 = y + real(k*a)*F + a*z2;
                T = t + k;
                F1 = f(T, y1);
                arma::Col<eT> z3(y.n_elem, arma::fill::zeros);

                res = std::numeric_limits<real>::infinity();

                for (u_long it=0; it < opts.solver_max_iter; ++it)
                {
                    arma::Col<eT> r = P *(z3 - k*F1);
                    r = arma::solve(arma::trimatl(L), r);
                    r = arma::solve(arma::trimatu(U), r);

                    res = arma::norm(r);

                    z3 -= r;
                    Y = y1 + eT(C/2)*z3;
                    F1 = f(T, Y);

                    if (res < tol/5)
                        break;
                }

                if (res > tol/5)
                    return std::make_tuple(false, real(0), arma::Col<eT>{}, arma::Col<eT>{});

                arma::Col<eT> err = eT(k*(C-1)/3)*F + eT(1.0/3.0)*z2 + eT(-C/3)*z3; // embedded error estimate
                err = arma::solve(arma::trimatl(L), P*err);
                err = arma::solve(arma::trimatu(U), err);
                res = arma::norm(err);
                
                return std::make_tuple(true, res, std::move(Y), std::move(F1));
            }
        };

        template <std::floating_point real, typename vec, std::invocable<real,vec> Func, typename Jacobian>
        ivpResults<real,vec> trbdf2(Func f, Jacobian jac, const std::vector<real>& tspan, const vec& y0, const ivpOpts<real>& opts = {}, const std::vector<Event<real,vec>>& events = {})
        {
            _ivp_helper::check_range<real>(tspan);

            auto T = tspan.begin();

            _ivp_helper::khan_stable_sum<real> _add;
            real t = *T;
            vec y = y0;
            vec F = f(t, y);
            real k = opts.initial_step;
            if (k <= 0)
                k = (tspan[1] - tspan[0]) * real(0.01);

            ivpResults<real,vec> sol(opts.dense_output);
            sol.t.push_back(t);
            sol.y.push_back(y);
            sol.f.push_back(F);

            ++T;
            for (; T != tspan.end(); ++T)
            {
                const real tf = *T;
                while (t < tf)
                {
                    if (std::abs(tf - (t + k)) < std::abs(tf+1)*std::numeric_limits<real>::epsilon())
                        k = tf - t;
                    else
                        k = std::min(k, tf - t);

                    __rk2_step<vec> rk2(t, y, std::forward<Func>(f), std::forward<Jacobian>(jac));
                    real tol = std::max<real>(opts.rtol*__vmath::norm_impl(y), opts.atol);

                    while (true)
                    {
                        auto [success, err, y1, F1] = rk2(t, y, F, k, std::forward<Func>(f), std::forward<Jacobian>(jac), opts);

                        if (not success) {
                            sol.flag = ExitFlag::NL_FAIL;
                            return sol;
                        }

                        real k1 = k * std::min<real>(10.0, std::max<real>(0.1, real(0.9)*std::sqrt(tol/err)));

                        if (k1 < (std::abs(t)+1)*std::numeric_limits<real>::epsilon()) {
                            sol.flag = ExitFlag::STEP_FAIL;
                            return sol;
                        }

                        if (err > tol) {
                            k = k1;
                            continue;
                        }

                        real alpha = 1.;
                        auto event_out = _ivp_helper::handle_event(t, y, t+k, y1, events, opts.event_tol);
                        if (event_out)
                            alpha = event_out->second;

                        if ((0 < alpha) and (alpha < 1))
                            k *= alpha;
                        else {
                            t = _add(t, k);
                            y = std::move(y1);
                            F = std::move(F1);
                            k = k1;

                            bool is_grid = std::abs(tf - t) < std::max<real>(1,t)*std::numeric_limits<real>::epsilon();
                            if (event_out or is_grid or opts.dense_output or !opts.grid_only) {
                                sol.t.push_back(t);
                                sol.y.push_back(y);
                                sol.f.push_back(F);

                                if (event_out) {
                                    sol.flag = ExitFlag::EVENT;
                                    return sol;
                                }
                            }

                            break;
                        }
                    } // one step
                } // between grid points in tspan
            } // tspan

            sol.flag = ExitFlag::SUCCESS;
            return sol;
        }
    
        template <std::floating_point real, typename vec, std::invocable<real,vec> Func>
        inline ivpResults<real,vec> trbdf2(Func f, const std::vector<real>& tspan, const vec& y0, const ivpOpts<real>& opts = {}, const std::vector<Event<real,vec>>& events = {})
        {
            return trbdf2(std::forward<Func>(f), int{}, tspan, y0, opts, events);
        }
    } // namespace ode
} // namespace numerics


#endif