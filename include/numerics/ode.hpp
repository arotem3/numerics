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
    enum class event_direction {
        NEGATIVE = -1,
        ALL = 0,
        POSITIVE = 1
    };

    // --- Utility ---------------- //
    arma::rowvec diffvec(const arma::vec& x, double x0, uint k=1);
    void diffmat(arma::sp_mat& D, const arma::vec& x, uint k=1, uint bdw=2);
    void diffmat(arma::mat& D, const arma::vec& x, uint k=1, uint bdw=2);
    void diffmat4(arma::mat& D, arma::vec& x, double L, double R, uint m);
    void diffmat2(arma::mat& D, arma::vec& x, double L, double R, uint m);
    void cheb(arma::mat& D, arma::vec& x, double L, double R, uint m);
    void cheb(arma::mat& D, arma::vec& x, uint m);
    // --- IVPs ------------------- //
    class ivp {
        protected:
        std::vector<event_direction> event_dirs;
        std::vector<std::function<double(double, const arma::rowvec&)>> events;
        double event_handle(double prev_t, const arma::rowvec& prev_U, double t, const arma::rowvec& V, double k);
        
        public:
        uint max_nonlin_iter;
        double max_nonlin_err;
        uint stopping_event;

        ivp() {
            max_nonlin_iter = 500;
            max_nonlin_err = 1e-8;
            stopping_event = -1;
        }

        void add_stopping_event(const std::function<double(double,const arma::rowvec&)>& event, event_direction dir = event_direction::ALL) {
            events.push_back(event);
            event_dirs.push_back(dir);
        }
    };

    class rk45 : public ivp {
        public:
        double adaptive_step_min;
        double adaptive_max_err;
        rk45(double tol = 1e-4) {
            adaptive_step_min = 1e-6;
            adaptive_max_err = tol;
        }
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U);
    };

    class rk45i : public ivp {
        public:
        double adaptive_step_min;
        double adaptive_max_err;
        rk45i(double tol = 1e-4) {
            adaptive_step_min = 1e-6;
            adaptive_max_err = tol;
        }
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U);
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                    const std::function<arma::mat(double,const arma::rowvec&)>& jacobian,
                    arma::vec& t, arma::mat& U);
    };

    class rk4 : public ivp {
        public:
        double step;
        rk4(double step_size = 0.1) {
            step = step_size;
        }
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U);
    };

    class rk5i : public ivp {
        public:
        double step;
        rk5i(double step_size = 0.1) {
            step = step_size;
        }
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U);
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                    const std::function<arma::mat(double,const arma::rowvec&)>& jacobian,
                    arma::vec& t, arma::mat& U);
    };

    class am1 : public ivp {
        public:
        double step;
        am1(double step_size = 0.1) {
            step = step_size;
        }
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U);
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                    const std::function<arma::mat(double, const arma::vec&)>& jacobian,
                    arma::vec& t, arma::mat& U);
    };

    class am2 : public ivp {
        public:
        double step;
        am2(double step_size = 0.1) {
            step = step_size;
        }
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U);
        void ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                    const std::function<arma::mat(double, const arma::rowvec&)>& jacobian,
                    arma::vec& t, arma::mat& U);
    };
    // --- BVPs ------------------- //

    class bvp {
        protected:
        int num_iter;

        public:
        uint max_iterations;
        double tol;
        int num_iterations() {
            return num_iter;
        }
    };

    class bvp_k : public bvp {
        private:
        int k;

        public:
        bvp_k(uint order = 4, double tolerance = 1e-5) {
            tol = tolerance;
            k = order;
            max_iterations = 100;
        }

        void ode_solve(arma::vec& x, arma::mat& U,
                    const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                    const std::function<arma::vec(const arma::rowvec&, const arma::rowvec&)>& bc);
        void ode_solve(arma::vec& x, arma::mat& U,
                    const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                    const std::function<arma::mat(double, const arma::rowvec&)>& jacobian,
                    const std::function<arma::vec(const arma::rowvec&, const arma::rowvec&)>& bc);
    };

    class bvp_cheb : public bvp {
        public:
        uint num_pts;
        bvp_cheb(uint num_points = 32, double tolerance = 1e-5) {
            tol = tolerance;
            num_pts = num_points;
            max_iterations = 100;
        }

        void ode_solve(arma::vec& x, arma::mat& U,
                    const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                    const std::function<arma::vec(const arma::rowvec&, const arma::rowvec&)>& bc,
                    const std::function<arma::rowvec(double)>& guess);
        void ode_solve(arma::vec& x, arma::mat& U,
                    const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                    const std::function<arma::mat(double, const arma::rowvec&)>& jacobian,
                    const std::function<arma::vec(const arma::rowvec&, const arma::rowvec&)>& bc,
                    const std::function<arma::rowvec(double)>& guess);
    };

    class bvpIIIa : public bvp {
        public:
        bvpIIIa(double tolerance = 1e-5) {
            tol = tolerance;
            max_iterations = 100;
            num_iter = 0;
        }
        
        void ode_solve(arma::vec& x, arma::mat& U,
                        const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                        const std::function<arma::vec(const arma::rowvec&,const arma::rowvec&)>& bc);
        void ode_solve(arma::vec& x, arma::mat& U,
                        const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
                        const std::function<arma::mat(double, const arma::rowvec&)>& jacobian,
                        const std::function<arma::vec(const arma::rowvec&,const arma::rowvec&)>& bc);
    };

    // --- PDEs -------------------- //
    class boundary_conditions_2d {
        public:
        double lower_x, upper_x, lower_y, upper_y;
        std::function<arma::mat(const arma::mat&, const arma::mat&)> dirichlet_condition;
    };

    void poisson2d(arma::mat& X, arma::mat& Y, arma::mat& U,
                   const std::function<arma::mat(const arma::mat&, const arma::mat&)>& f,
                   const boundary_conditions_2d& bc, int m = 32);
}