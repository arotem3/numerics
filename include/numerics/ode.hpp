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

namespace ode {
    // --- IVP events ------------ //
    typedef enum EVENT_DIR {
        NEGATIVE = -1,
        ALL = 0,
        POSITIVE = 1
    } event_direction;

    struct event_out {
        event_direction dir;
        double val;
    };

    typedef std::function<event_out(double t, const arma::rowvec&)> event_func;

    // --- Utility ---------------- //
    void diffmat4(arma::mat& D, arma::vec& x, double L, double R, uint m);
    void diffmat2(arma::mat& D, arma::vec& x, double L, double R, uint m);
    void cheb(arma::mat& D, arma::vec& x, double L, double R, uint m);
    void cheb(arma::mat& D, arma::vec& x, uint m);
    // --- IVPs ------------------- //
    class ivp {
        protected:
        uint max_nonlin_iter;
        double max_nonlin_err;
        double adaptive_step_min;
        double adaptive_step_max;
        double adaptive_max_err;
        double step;
        uint stopping_event;
        std::vector<event_func> events;
        bool specify_jacobian;

        double event_handle(double prev_t, const arma::rowvec& prev_U, double t, const arma::rowvec& V, double k);
        
        public:
        ivp() {
            max_nonlin_iter = 50;
            max_nonlin_err = 1e-6;
            adaptive_step_min = 1e-3;
            adaptive_step_max = 0.5;
            adaptive_max_err = 1e-4;
            step = 1e-1;
            stopping_event = -1;
            specify_jacobian = false;
        }
    };

    class rk45 : public ivp {
        public:
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U);
    };

    class bdf23 : public ivp {
        public:
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U);
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                    const std::function<arma::mat(double, const arma::rowvec&)>& jacobian,
                    arma::vec& t, arma::mat& U);
    };

    class rk4 : public ivp {
        public:
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U);
    };

    class rk5i : public ivp {
        public:
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U);
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                    const std::function<arma::mat(double,const arma::vec&)>& jacobian,
                    arma::vec& t, arma::mat& U);
    };

    class am1 : public ivp {
        public:
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U);
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                    const std::function<arma::mat(double, const arma::vec&)>& jacobian,
                    arma::vec& t, arma::mat& U);
    };

    class am2 : public ivp {
        public:
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U);
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                    const std::function<arma::mat(double, const arma::rowvec&)>& jacobian,
                    arma::vec& t, arma::mat& U);
    };
    // --- BVPs ------------------- //
    typedef enum BVP_SOLVERS {
        FOURTH_ORDER,
        SECOND_ORDER,
        CHEBYSHEV
    } bvp_solvers;

    class boundary_conditions {
        public:
        double xL, xR;
        std::function<arma::mat(const arma::rowvec& uL, const arma::rowvec& uR)> condition;
    };

    class bvp {
        private:
        int num_iter;

        public:
        uint num_points, max_iterations;
        double tol;
        bvp_solvers order;
        int num_iterations() {
            return num_iter;
        }
        
        bvp(int N = 32) : num_points(N) {
            order = FOURTH_ORDER;
            tol = 1e-5;
            max_iterations = 100;
        }

        void ode_solve(arma::vec& x, arma::mat& U,
                    const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                    const boundary_conditions& bc,
                    const std::function<arma::vec(const arma::vec&)>& guess);
        void ode_solve(arma::vec& x, arma::mat& U,
                    const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                    const std::function<arma::mat(double, const arma::rowvec&)>& jacobian,
                    const boundary_conditions& bc,
                    const std::function<arma::vec(const arma::vec&)>& guess);
    };
}