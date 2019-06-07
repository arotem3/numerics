#pragma once

#include "numerics.hpp"

/* Copyright 2019 Amit Rotem

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

namespace ODE {
    // --- ode solver constants --- //
        const double rk45_kmin = 1.0e-4;
        const double rk45_kmax = 0.5;
        const double rk45_qmin = 1e-2;
        const double rk45_qmax = 10;
        const double implicit_err = 1e-6;

        const int implicit_ode_max_iter = 400;
    // --- enumerators ------------ //
        typedef enum ODE_SOLVER {
            RK45,
            BDF23,
            RK4,
            RK5I,
            AM1,
            AM2
        } ode_solver;

        typedef enum BVP_SOLVERS {
            FOURTH_ORDER,
            SECOND_ORDER,
            CHEBYSHEV
        } bvp_solvers;

        typedef enum EVENT_DIR {
            NEGATIVE = -1,
            ALL = 0,
            POSITIVE = 1
        } event_direction;
    // --- input objects ---------- //
        typedef std::function<arma::rowvec(double, const arma::rowvec&)> odefun;

        typedef std::function<arma::mat(double, const arma::rowvec&)> odeJac;

        typedef std::function<arma::vec(const arma::vec&,const arma::vec&)> pde2fun;

        typedef std::function<arma::mat(const arma::vec&)> soln_init;

        typedef struct BC2D {
            double lower_x, lower_y, upper_x, upper_y;
            std::function<arma::vec(const arma::vec&)> lower_x_bc, lower_y_bc, upper_x_bc, upper_y_bc;

            BC2D() {
                lower_x = -1;
                lower_y = -1;
                upper_x =  1;
                upper_y =  1;
                lower_x_bc = [](const arma::vec& x) -> arma::vec {return arma::zeros(arma::size(x));};
                lower_y_bc = lower_x_bc;
                upper_x_bc = lower_x_bc;
                upper_y_bc = lower_x_bc;
            }
        } bcfun_2d;

        typedef struct BCFUN {
            double xL;
            double xR;
            std::function<arma::rowvec(const arma::rowvec&,const arma::rowvec&)> func;
        } bcfun;
    // --- output objects --------- //
        typedef struct SOLUTION_2D {
            arma::mat X, Y, U;
            void save(std::ostream& out) {
                out << X.n_rows << " " << X.n_cols << std::endl;
                out.precision(12);
                X.raw_print(out);
                Y.raw_print(out);
                U.raw_print(out);
            }
            void load(std::istream& in) {
                int n, m;
                in >> n >> m;
                X = arma::zeros(n,m);
                Y = arma::zeros(n,m);
                U = arma::zeros(n,m);
                for (int i(0); i < n; ++i) {
                    for (int j(0); j < m; ++j) {
                        in >> X(i,j);
                    }
                }
                for (int i(0); i < n; ++i) {
                    for (int j(0); j < m; ++j) {
                        in >> Y(i,j);
                    }
                }
                for (int i(0); i < n; ++i) {
                    for (int j(0); j < m; ++j) {
                        in >> U(i,j);
                    }
                }
            }
        } soln_2d;

        typedef struct DSOLNP {
            arma::vec independent_var_values;
            arma::mat solution_values;
            numerics::polyInterp soln;
        } dsolnp;

        typedef numerics::CubicInterp dsolnc;
    // --- options objects -------- //
        typedef struct IVP_EVENT_OUT {
            event_direction dir;
            double val;
        } event_out;

        typedef std::function<event_out(double t, const arma::rowvec&)> event_func;

        typedef struct IVP_OPTIONS {
            // inputs
            uint max_nonlin_iter;
            double max_nonlin_err;
            double adaptive_step_min;
            double adaptive_step_max;
            double adaptive_max_err;
            double step;
            uint stopping_event;
            std::vector<event_func> events;
            odeJac* ode_jacobian;

            IVP_OPTIONS() {
                max_nonlin_iter = 100;
                max_nonlin_err = 1e-8;
                step  = 1e-2;
                adaptive_max_err = 0;
                adaptive_step_max = 0;
                adaptive_step_min = 0;
                stopping_event = -1;
                ode_jacobian = nullptr;
                num_FD_approx_needed = 0;
                num_nonlin_iters_returned = 0;
            }
            void standard_adaptive() {
                adaptive_step_min = rk45_kmin;
                adaptive_step_max = rk45_kmax;
                adaptive_max_err = 1e-4;
            }

            // outputs
            uint num_FD_approx_needed;
            uint num_nonlin_iters_returned;
        } ivp_options;

        double event_handle(ivp_options& opts, double prev_t, const arma::rowvec& prev_U, double t, const arma::rowvec& V, double k);

        typedef struct NONLIN_BVP_OPTS {
            // inputs
            uint num_points;
            numerics::nonlin_opts nlnopts;
            bvp_solvers order;
            odeJac* jacobian_func;

            NONLIN_BVP_OPTS() {
                num_points = 30;
                order = bvp_solvers::FOURTH_ORDER;
                jacobian_func = nullptr;
            }
        } bvp_opts;
    // --- Utility ---------------- //
        void diffmat4(arma::mat& D, arma::vec& x, double L, double R, uint m);
        void diffmat2(arma::mat& D, arma::vec& x, double L, double R, uint m);
        void cheb(arma::mat& D, arma::vec& x, double L, double R, uint m);
        void cheb(arma::mat& D, arma::vec& x, uint m);
    // --- IVPs ------------------- //
        void rk45(const odefun&, arma::vec&, arma::mat&, ivp_options&);
        ivp_options rk45(const odefun&, arma::vec&, arma::mat&);
        arma::vec rk45(std::function<double(double,double)>, arma::vec&, double, ivp_options&);
        arma::vec rk45(std::function<double(double,double)>, arma::vec&, double);

        void bdf23(const odefun&, arma::vec&, arma::mat&, ivp_options&);
        ivp_options bdf23(const odefun&, arma::vec&, arma::mat&);
        arma::vec bdf23(std::function<double(double,double)>, arma::vec&, double, ivp_options&);
        arma::vec bdf23(std::function<double(double,double)>, arma::vec&, double);

        void rk4(const odefun&, arma::vec&, arma::mat&, ivp_options&);
        ivp_options rk4(const odefun&, arma::vec&, arma::mat&);
        arma::vec rk4(std::function<double(double,double)>, arma::vec&, double, ivp_options&);
        arma::vec rk4(std::function<double(double,double)>, arma::vec&, double);

        void rk5i(const odefun&, arma::vec&, arma::mat&, ivp_options&);
        ivp_options rk5i(const odefun&, arma::vec&, arma::mat&);
        arma::vec rk5i(std::function<double(double,double)>, arma::vec&, double, ivp_options&);
        arma::vec rk5i(std::function<double(double,double)>, arma::vec&, double);

        void am1(const odefun&, arma::vec&, arma::mat&, ivp_options&);
        ivp_options am1(const odefun&, arma::vec&, arma::mat&);
        arma::vec am1(std::function<double(double,double)>, arma::vec&, double, ivp_options&);
        arma::vec am1(std::function<double(double,double)>, arma::vec&, double);

        void am2(const odefun&, arma::vec&, arma::mat&, ivp_options&);
        ivp_options am2(const odefun&, arma::vec&, arma::mat&);
        arma::vec am2(std::function<double(double,double)>, arma::vec&, double, ivp_options&);
        arma::vec am2(std::function<double(double,double)>, arma::vec&, double);

        dsolnc ivp(const odefun&, arma::vec&, arma::mat&, ivp_options&, ode_solver solver = RK45);
        dsolnc ivp(const odefun&, arma::vec&, arma::mat&, ode_solver solver = RK45);
    // --- BVPs ------------------- //
        dsolnp bvp(const odefun&, const bcfun&, const soln_init&, bvp_opts&);
        dsolnp bvp(const odefun&, const bcfun&, const soln_init&);
    // --- PDEs ------------------- //
        soln_2d poisson2d(const pde2fun&, const bcfun_2d&, uint num_pts = 48);
}