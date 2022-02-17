#ifndef NUMERICS_ODE_RK34I
#define NUMERICS_ODE_RK34I

#include "numerics/ode/ode_base.hpp"
#include "numerics/derivatives.hpp"
#include "numerics/optimization/newton.hpp"
#include "numerics/optimization/fzero.hpp"

namespace numerics
{
    namespace ode
    {
        template <std::floating_point real, typename vec>
        struct rk34iResults : public ivpResults<real,vec>
        {
            template <std::floating_point T, typename V, std::invocable<T,V> Func, typename Jacobian>
            friend rk34iResults<T,V> rk34i(Func, Jacobian, const std::vector<T>&, const V&, const ivpOpts<T>&, const std::vector<Event<T,V>>&);

        public:
            std::vector<vec> y_half;

        public:
            rk34iResults(bool dense=false) : ivpResults<real,vec>(dense) {}

            // C1 interpolation using quartic hermite polynomial
            vec operator()(real s) const override
            {
                if (not this->dense)
                    throw std::runtime_error("rk45Results operator() error: solution is not dense, cannot evaluate.");

                if (s == this->t.front())
                    return this->y.front();
                else if (s == this->t.back())
                    return this->y.back();
                else if (s < this->t.front() || this->t.back() < s)
                    throw std::runtime_error("interpolation error: t = " + std::to_string(s) + " outside of domain (" + std::to_string(this->t.front()) + ", " + std::to_string(this->t.back()) + ").");


                size_t i = std::distance(this->t.begin(), std::lower_bound(this->t.begin(), this->t.end(), s));                
                if (i > 0)
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
                // vec u = b0*this->y[i] + b1*this->f[i] + b2*this->y_half[i] + b3*this->f[i+1] + b4*this->y[i+1];
                return u;
            }

            std::vector<vec> operator()(std::vector<real> s) const override
            {
                if (not this->dense)
                    throw std::runtime_error("rk34i operator() error: solution is not dense, cannot evaluate.");
            
                std::sort(s.begin(), s.end());
                std::vector<vec> u;
                u.reserve(s.size());

                _ivp_helper::check_interp_range(s, this->t);

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

        template <typename vec>
        class __rk34i_step
        {
        public:
            template <std::floating_point real, std::invocable<real,vec> Func, typename Dummy>
            __rk34i_step(real t, const vec& y, Func f, Dummy jac_dummy) {}

            template <std::floating_point real, std::invocable<real,vec> Func>
            std::tuple<bool,real,vec,vec,vec> operator()(real t, const vec& y, const vec& F, real k, Func f, int jac_dummy, const ivpOpts<real>& opts)
            {
                static constexpr real rk4c[] = {0.25, 0.75, 0.55, 0.5, 1.0};
                static constexpr real rk4a[4][4] = {
                    {0.5, 0, 0, 0},
                    {0.34, -0.04, 0, 0},
                    {0.2727941176470588, -0.05036764705882353, 0.02757352941176471, 0},
                    {1.041666666666667, -1.020833333333333, 7.8125, -7.083333333333333}
                };
                static constexpr real rk4e[] = {-0.1875, -0.84375, 0.78125, 0.0, 0.25};
                static constexpr real denseb[] = {0.8402777777777778, -0.4479166666666667, 3.616898148148148, -3.541666666666667, 0.03240740740740741};

                vec z[5];
                vec Y, F1;

                optimization::OptimizationOptions<real> optim_opts;
                optim_opts.ftol = opts.rtol / 5;
                optim_opts.max_iter = opts.solver_max_iter;

                #pragma unroll
                for (int i=0; i < 5; ++i)
                {
                    real s = t + rk4c[i]*k;
                    vec p = y;
                    
                    #pragma unroll
                    for (int j=0; j < i; ++j)
                        p += rk4a[i-1][j]*z[j];
                        
                    auto res = [&](const vec& u) -> vec
                    {
                        Y = p + real(0.25)*u;
                        F1 = f(s, Y);
                        return u - k*F1;
                    };

                    z[i] = 0*y;
                    optimization::OptimizationResults<vec> rslts = optimization::newton(z[i], res, optim_opts);
                    
                    if (rslts.flag != optimization::ExitFlag::CONVERGED)
                        return std::make_tuple(false, real(0), vec{}, vec{}, vec{});
                }

                vec err = rk4e[0]*z[0] + rk4e[1]*z[1] + rk4e[2]*z[2] + rk4e[4]*z[4];
                vec y_half = y + denseb[0]*z[0] + denseb[1]*z[1] + denseb[2]*z[2] + denseb[3]*z[3] + denseb[4]*z[4];
                real abs_err = __vmath::norm_impl(err);

                return std::make_tuple(true, abs_err, std::move(Y), std::move(F1), std::move(y_half));
            }
        };

        template <scalar_field_type T>
        class __rk34i_step<T>
        {
        public:
            template <std::floating_point real, std::invocable<real,T> Func, typename Dummy>
            __rk34i_step(real t, const T& y, Func f, Dummy jac_dummy) {}

            template <std::floating_point real, std::invocable<real,T> Func>
            std::tuple<bool,real,T,T,T> operator()(real t, const T& y, const T& F, real k, Func f, int jac_dummy, const ivpOpts<real>& opts)
            {
                static constexpr real rk4c[] = {0.25, 0.75, 0.55, 0.5, 1.0};
                static constexpr real rk4a[4][4] = {
                    {0.5, 0, 0, 0},
                    {0.34, -0.04, 0, 0},
                    {0.2727941176470588, -0.05036764705882353, 0.02757352941176471, 0},
                    {1.041666666666667, -1.020833333333333, 7.8125, -7.083333333333333}
                };
                static constexpr real rk4e[] = {-0.1875, -0.84375, 0.78125, 0.0, 0.25};
                static constexpr real denseb[] = {0.8402777777777778, -0.4479166666666667, 3.616898148148148, -3.541666666666667, 0.03240740740740741};

                T z[5];
                T Y, F1;

                real tol = std::max<real>(opts.atol, opts.rtol*std::abs(y)) / 5;

                #pragma unroll
                for (int i=0; i < 5; ++i)
                {
                    real s = t + rk4c[i]*k;
                    T p = y;
                    
                    #pragma unroll
                    for (int j=0; j < i; ++j)
                        p += rk4a[i-1][j]*z[j];
                        
                    auto res = [&](const T& u) -> T
                    {
                        Y = p + T(0.25)*u;
                        F1 = f(s, Y);
                        return u - k*F1;
                    };

                    z[i] = optimization::newton_1d(res, T(0.0), tol);
                    
                    if (std::abs(res(z[i])) > tol)
                        return std::make_tuple(false, real(0), T{}, T{}, T{});
                }

                T err = rk4e[0]*z[0] + rk4e[1]*z[1] + rk4e[2]*z[2] + rk4e[4]*z[4];
                T y_half = y + denseb[0]*z[0] + denseb[1]*z[1] + denseb[2]*z[2] + denseb[3]*z[3] + denseb[4]*z[4];
                real abs_err = std::abs(err);

                return std::make_tuple(true, abs_err, Y, F1, y_half);
            }

            template <std::floating_point real, std::invocable<real,T> Func, std::invocable<real,T> Jacobian>
            std::tuple<bool,real,T,T,T> operator()(real t, const T& y, const T& F, real k, Func f, Jacobian jacobian, const ivpOpts<real>& opts)
            {
                static constexpr real rk4c[] = {0.25, 0.75, 0.55, 0.5, 1.0};
                static constexpr real rk4a[4][4] = {
                    {0.5, 0, 0, 0},
                    {0.34, -0.04, 0, 0},
                    {0.2727941176470588, -0.05036764705882353, 0.02757352941176471, 0},
                    {1.041666666666667, -1.020833333333333, 7.8125, -7.083333333333333}
                };
                static constexpr real rk4e[] = {-0.1875, -0.84375, 0.78125, 0.0, 0.25};
                static constexpr real denseb[] = {0.8402777777777778, -0.4479166666666667, 3.616898148148148, -3.541666666666667, 0.03240740740740741};

                T z[5];
                T Y, F1;

                real tol = std::max<real>(opts.atol, opts.rtol*std::abs(y)) / 5;

                #pragma unroll
                for (int i=0; i < 5; ++i)
                {
                    real s = t + rk4c[i]*k;
                    T p = y;
                    
                    #pragma unroll
                    for (int j=0; j < i; ++j)
                        p += rk4a[i-1][j]*z[j];
                        
                    auto res = [&](const T& u) -> T
                    {
                        Y = p + T(0.25)*u;
                        F1 = f(s, Y);
                        return u - k*F1;
                    };

                    auto d_res = [&](const T& u) -> T
                    {
                        T J = jacobian(s, Y);
                        return T(1.0) - (k/4)*J;
                    };

                    z[i] = optimization::newton_1d(res, d_res, T(0.0), tol);
                    
                    if (std::abs(res(z[i])) > tol)
                        return std::make_tuple(false, real(0), T{}, T{}, T{});
                }

                T err = rk4e[0]*z[0] + rk4e[1]*z[1] + rk4e[2]*z[2] + rk4e[4]*z[4];
                T y_half = y + denseb[0]*z[0] + denseb[1]*z[1] + denseb[2]*z[2] + denseb[3]*z[3] + denseb[4]*z[4];
                real abs_err = std::abs(err);

                return std::make_tuple(true, abs_err, Y, F1, y_half);
            }
        };

        #ifdef NUMERICS_WITH_ARMA
        template <scalar_field_type eT>
        class __rk34i_step<arma::Col<eT>>
        {
        private:
            arma::Mat<eT> J;
        
        public:
            template <std::floating_point real, std::invocable<real,arma::Col<eT>> Func, std::invocable<real,arma::Col<eT>> Jacobian>
            __rk34i_step(real t, const arma::Col<eT>& y, Func f, Jacobian jacobian)
            {
                J = jacobian(t, y);
            }

            template <std::floating_point real, std::invocable<real,arma::Col<eT>> Func>
            __rk34i_step(real t, const arma::Col<eT>& y, Func f, int jac_dummy)
            {
                auto F = [&](const arma::Col<eT>& u) -> arma::Col<eT>
                {
                    return f(t, u);
                };

                J = jacobian(F, y);
            }

            template <std::floating_point real, std::invocable<real,arma::Col<eT>> Func, typename Jacobian>
            std::tuple<bool,real,arma::Col<eT>,arma::Col<eT>,arma::Col<eT>> operator()(real t, const arma::Col<eT>& y, const arma::Col<eT>& F, real k, Func f, Jacobian jac_dummy, const ivpOpts<real>& opts)
            {
                static constexpr real rk4c[] = {0.25, 0.75, 0.55, 0.5, 1.0};
                static constexpr real rk4a[4][4] = {
                    {0.5, 0, 0, 0},
                    {0.34, -0.04, 0, 0},
                    {0.2727941176470588, -0.05036764705882353, 0.02757352941176471, 0},
                    {1.041666666666667, -1.020833333333333, 7.8125, -7.083333333333333}
                };
                static constexpr real rk4e[] = {-0.1875, -0.84375, 0.78125, 0.0, 0.25};
                static constexpr real denseb[] = {0.8402777777777778, -0.4479166666666667, 3.616898148148148, -3.541666666666667, 0.03240740740740741};

                arma::Mat<eT> A = eT(-k/4)*J;
                A.diag() += eT(1.0);
                arma::Mat<eT> L, U, P;
                bool successful_decomp = arma::lu(L, U, P, A);
                if (not successful_decomp)
                    return std::make_tuple(false, real(0), arma::Col<eT>{}, arma::Col<eT>{}, arma::Col<eT>{});

                arma::Col<eT> z[5];
                arma::Col<eT> Y, F1;

                real tol = std::max<real>(opts.atol, opts.rtol*arma::norm(y)) / 5;

                #pragma unroll
                for (int i=0; i < 5; ++i)
                {
                    real s = t + rk4c[i]*k;
                    arma::Col<eT> p = y;
                    
                    #pragma unroll
                    for (int j=0; j < i; ++j)
                        p += rk4a[i-1][j]*z[j];
                        
                    z[i] = arma::zeros<arma::Col<eT>>(y.n_elem);
                    F1 = f(s, p);

                    real res = std::numeric_limits<real>::infinity();
                    for (u_long it=0; it < opts.solver_max_iter; ++it)
                    {
                        arma::Col<eT> r = P * (z[i] - k * F1);
                        r = arma::solve(arma::trimatl(L), r);
                        r = arma::solve(arma::trimatu(U), r);

                        res = arma::norm(r);

                        z[i] -= r;
                        Y = p + eT(0.25)*z[i];
                        F1 = f(s, Y);

                        if (res < tol)
                            break;
                    }
                    
                    if (res > tol)
                        return std::make_tuple(false, real(0), arma::Col<eT>{}, arma::Col<eT>{}, arma::Col<eT>{});
                }

                arma::Col<eT> err = rk4e[0]*z[0] + rk4e[1]*z[1] + rk4e[2]*z[2] + rk4e[4]*z[4];
                err = arma::solve(arma::trimatl(L), P*err);
                err = arma::solve(arma::trimatu(U), err);
                real abs_err = arma::norm(err);

                arma::Col<eT> y_half = y + denseb[0]*z[0] + denseb[1]*z[1] + denseb[2]*z[2] + denseb[3]*z[3] + denseb[4]*z[4];

                return std::make_tuple(true, abs_err, std::move(Y), std::move(F1), std::move(y_half));
            }
        };
        #endif

        template <std::floating_point real, typename vec, std::invocable<real,vec> Func, typename Jacobian>
        rk34iResults<real,vec> rk34i(Func f, Jacobian jac, const std::vector<real>& tspan, const vec& y0, const ivpOpts<real>& opts = {}, const std::vector<Event<real,vec>>& events = {})
        {
            _ivp_helper::check_range<real>(tspan);

            auto T = tspan.begin();

            _ivp_helper::khan_stable_sum<real> _add;
            real t = *T;
            vec y = y0;
            vec F = f(t, y);

            real k = (opts.initial_step <= 0) ? real(0.01)*(tspan[1] - tspan[0]) : opts.initial_step;
            const real max_step = (opts.max_step <= 0) ? real(0.1)*(tspan.back() - tspan.front()) : opts.max_step;

            rk34iResults<real,vec> sol(opts.dense_output);
            sol.t.push_back(t);
            sol.y.push_back(y);
            sol.f.push_back(F);

            ++T;
            for (; T != tspan.end(); ++T)
            {
                const real tf = *T;
                while (t < tf)
                {
                    __rk34i_step<vec> step(t, y, std::forward<Func>(f), std::forward<Jacobian>(jac));
                    real tol = std::max<real>(opts.rtol*__vmath::norm_impl(y), opts.atol);

                    while (true)
                    {
                        if (std::abs(tf - (t + k)) < std::abs(tf+1)*std::numeric_limits<real>::epsilon())
                            k = tf - t;
                        else
                            k = std::min(k, tf - t);

                        auto [success, err, y1, F1, y_half] = step(t, y, F, k, std::forward<Func>(f), std::forward<Jacobian>(jac), opts);

                        if (not success) {
                            sol.flag = ExitFlag::NL_FAIL;
                            return sol;
                        }

                        real k1 = k * std::min<real>(10.0, std::max<real>(0.1, real(0.9)*std::pow(tol/err, real(0.25))));
                        k1 = std::min<real>(k1, max_step);

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
                            if (opts.dense_output)
                                sol.y_half.push_back(std::move(y_half));

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
        inline rk34iResults<real,vec> rk34i(Func f, const std::vector<real>& tspan, const vec& y0, const ivpOpts<real>& opts = {}, const std::vector<Event<real,vec>>& events = {})
        {
            return rk34i(std::forward<Func>(f), int{}, tspan, y0, opts, events);
        }
    } // namespace ode
} // namespace numerics


#endif