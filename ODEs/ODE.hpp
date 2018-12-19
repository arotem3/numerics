#pragma once

#include "../numerics.hpp"

namespace ODE {
    // --- ode solver constants
        const double rk45_kmin = 1.0e-4;
        const double rk45_kmax = 0.5;
        const double rk45_qmin = 1e-2;
        const double rk45_qmax = 10;
        const double implicit_err = 1e-6;

        const int implicit_ode_max_iter = 400;

        typedef enum ODE_SOLVER {
            RK45,
            BDF23,
            RK4,
            RK5I,
            AM1,
            AM2
        } ode_solver;

        typedef enum EVENT_DIR {
            NEGATIVE = -1,
            ALL = 0,
            POSITIVE = 1
        } event_direction;

        typedef struct IVP_EVENT_OUT {
            event_direction dir;
            double val;
        } event_out;

        typedef std::function<event_out(double t, const arma::rowvec&)> event_func;
        typedef std::function<arma::rowvec(double, const arma::rowvec&)> odefun;
        typedef struct BCFUN {
            double xL;
            double xR;
            std::function<arma::rowvec(const arma::rowvec&,const arma::rowvec&)> func;
        } bcfun;
        typedef std::function<arma::mat(const arma::vec&)> soln_init;

        typedef struct DSOLNP {
            arma::vec independent_var_values;
            arma::mat solution_values;
            numerics::polyInterp soln;
        } dsolnp;
        typedef numerics::CubicInterp dsolnc;
    
    // --- Utility --- //
        void cheb(arma::mat& D, arma::vec& x, double L, double R, size_t m);
    // --- IVPs --- //
        typedef struct IVP_OPTIONS {
            // inputs
            size_t max_nonlin_iter;
            double max_nonlin_err;
            double adaptive_step_min;
            double adaptive_step_max;
            double adaptive_max_err;
            double step;
            size_t stopping_event;
            std::vector<event_func> events;
            std::function<arma::mat(double,const arma::rowvec&)>* ode_jacobian;

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
            size_t num_FD_approx_needed;
            size_t num_nonlin_iters_returned;
        } ivp_options;

        double event_handle(ivp_options& opts, double prev_t, const arma::rowvec& prev_U, double t, const arma::rowvec& V, double k);
        
        void rk45(odefun, arma::vec&, arma::mat&, ivp_options&);
        ivp_options rk45(odefun, arma::vec&, arma::mat&);
        arma::vec rk45(std::function<double(double,double)>, arma::vec&, double, ivp_options&);
        arma::vec rk45(std::function<double(double,double)>, arma::vec&, double);

        void bdf23(odefun, arma::vec&, arma::mat&, ivp_options&);
        ivp_options bdf23(odefun, arma::vec&, arma::mat&);
        arma::vec bdf23(std::function<double(double,double)>, arma::vec&, double, ivp_options&);
        arma::vec bdf23(std::function<double(double,double)>, arma::vec&, double);

        void rk4(odefun, arma::vec&, arma::mat&, ivp_options&);
        ivp_options rk4(odefun, arma::vec&, arma::mat&);
        arma::vec rk4(std::function<double(double,double)>, arma::vec&, double, ivp_options&);
        arma::vec rk4(std::function<double(double,double)>, arma::vec&, double);

        void rk5i(odefun, arma::vec&, arma::mat&, ivp_options&);
        ivp_options rk5i(odefun, arma::vec&, arma::mat&);
        arma::vec rk5i(std::function<double(double,double)>, arma::vec&, double, ivp_options&);
        arma::vec rk5i(std::function<double(double,double)>, arma::vec&, double);

        void am1(odefun, arma::vec&, arma::mat&, ivp_options&);
        ivp_options am1(odefun, arma::vec&, arma::mat&);
        arma::vec am1(std::function<double(double,double)>, arma::vec&, double, ivp_options&);
        arma::vec am1(std::function<double(double,double)>, arma::vec&, double);

        void am2(odefun, arma::vec&, arma::mat&, ivp_options&);
        ivp_options am2(odefun, arma::vec&, arma::mat&);
        arma::vec am2(std::function<double(double,double)>, arma::vec&, double, ivp_options&);
        arma::vec am2(std::function<double(double,double)>, arma::vec&, double);

        numerics::CubicInterp IVP_solve(odefun, arma::vec&, arma::mat&, ivp_options&, ode_solver solver = RK45);
        numerics::CubicInterp IVP_solve(odefun, arma::vec&, arma::mat&, ode_solver solver = RK45);
    // --- BVPs --- //
        typedef enum BVP_SOLVERS {
            FOURTH_ORDER,
            SECOND_ORDER,
            CHEBYSHEV
        } bvp_solvers;

        class linear_BVP {
            //--- u''(x) = a(x) + b(x)*u(x) + c(x)*u'(x) ---//
            //----- L <= x <= R ----------------------------//
            //----- aL*u(L) + bL*u'(L) == gL ---------------//
            //----- aR*u(R) + bR*u'(R) == gR ---------------//
            private:
            double xL, xR, alphaL, alphaR, betaL, betaR, gammaL, gammaR;
            std::function<double(double)> a, b, c;
            void solve2(arma::vec&, arma::mat&, size_t);
            void solve4(arma::vec&, arma::mat&, size_t);
            void spectral_solve(arma::vec&, arma::mat&, size_t);

            public:
            linear_BVP();
            void set_boundaries(double, double);
            void set_RBC(double,double,double);
            void set_LBC(double,double,double);
            void set_a(std::function<double(double)>);
            void set_a(double);
            void set_b(std::function<double(double)>);
            void set_b(double);
            void set_c(std::function<double(double)>);
            void set_c(double);
            void solve(arma::vec&, arma::mat&, size_t, bvp_solvers solver = FOURTH_ORDER);
            dsolnp solve(size_t);
        };

        typedef struct NONLIN_BVP_OPTS {
            // inputs
            size_t num_points;
            numerics::lsqr_opts lsqropts;
            numerics::nonlin_opts nlnopts;
            numerics::nonlin_solver solver;

            NONLIN_BVP_OPTS() {
                num_points = 30;
                solver = numerics::BROYD;
            }
        } bvp_opts;

        dsolnp bvp(odefun, bcfun, soln_init, bvp_opts&);
        dsolnp bvp(odefun, bcfun, soln_init);
}