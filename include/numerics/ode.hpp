#ifndef NUMERICS_ODE_HPP
#define NUMERICS_ODE_HPP

namespace ode {
    typedef std::function<arma::vec(double,const arma::vec&)> odefunc;
    typedef std::function<arma::mat(double,const arma::vec&)> odejacobian;
    typedef std::function<arma::vec(const arma::vec&, const arma::vec&)> boundary_conditions;

    class ODESolution {
        friend class am1;
        friend class am2;
        friend class rk4;
        friend class rk45;
        friend class rk45i;
        friend class rk5i;
        friend class BVPk;
        friend class BVPCheb;
        friend class BVP3a;
        protected:
        u_long _dim;
        std::vector<double> _tvec;
        std::vector<arma::vec> _Uvec;
        arma::vec _t;
        arma::mat _U;
        int _flag;

        void _prepare() {
            if (_t.is_empty()) {
                _t = arma::conv_to<arma::vec>::from(_tvec);
                _U.set_size(_Uvec.size(), _dim);
                for (u_long i=0; i < _Uvec.size(); ++i) {
                    _U.row(i) = _Uvec.at(i).as_row();
                }
            } else {
                _tvec.clear(); _Uvec.clear();
                for (u_long i=0; i < _t.n_elem; ++i) {
                    _tvec.push_back(_t(i));
                    _Uvec.push_back(_U.row(i).as_col());
                }
            }
        }

        public:
        const arma::vec& t;
        const arma::mat& solution;
        const std::vector<double>& tvec;
        const std::vector<arma::vec>& solvec;
        const int& exit_flag;

        explicit ODESolution(u_long dim) : t(_t), solution(_U), tvec(_tvec), solvec(_Uvec), exit_flag(_flag) {
            if (dim == 0) throw std::runtime_error("ODESolution: require solution dimension (=" + std::to_string(dim) + ") > 0");
            _dim = dim;
            _flag = 0;
        }

        std::string get_exit_flag() {
            std::string flag;
            if (_flag == 0) {
                flag = "solution successfully found over the specified domain.";
            } else if (_flag == 1) {
                flag = "solution could not be found within specified error tolerance.";
            } else if (_flag == 2) {
                flag = "NaN or infinite value encountered.";
            } else if (_flag == 3) {
                flag = "could not solve system of linear equations.";
            }
            return flag;
        }
    };
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
    class InitialValueProblem {
        protected:
        long _stopping_event;
        std::vector<event_direction> event_dirs;
        std::vector<std::function<double(double, const arma::vec&)>> events;
        double event_handle(double prev_t, const arma::vec& prev_U, double t, const arma::vec& V, double k);
        
        void _check_range(double t0, double tf) {
            if (tf <= t0) throw std::runtime_error("(" + std::to_string(t0) + ", " + std::to_string(tf) + ") does not define a valid interval");
        }
        public:
        const long& stopping_event;

        InitialValueProblem() : stopping_event(_stopping_event) {
            _stopping_event = -1;
        }

        void add_stopping_event(const std::function<double(double,const arma::vec&)>& event, event_direction dir = event_direction::ALL) {
            events.push_back(event);
            event_dirs.push_back(dir);
        }

        virtual ODESolution ode_solve(const odefunc& f, double t0, double tf, const arma::vec& U0) = 0;
    };

    class AdaptiveIVP {
        protected:
        double _step_min;
        double _max_err;

        public:
        const double& step_min;
        const double& max_err;

        explicit AdaptiveIVP(double tol, double minstep) : step_min(_step_min), max_err(_max_err) {
            _step_min = minstep;
            _max_err = tol;
        }
    };

    class ImplicitIVP {
        protected:
        u_long _max_solver_iter;
        double _max_solver_err;

        public:
        const u_long& max_solver_iter;
        const double& max_solver_err;

        ImplicitIVP() : max_solver_iter(_max_solver_iter), max_solver_err(_max_solver_err) {
            _max_solver_iter = 500;
            _max_solver_err = 1e-8;
        }

        void set_solver_parameters(double t, long m) {
            _max_solver_err = t;
            _max_solver_iter = m;
        }
        virtual ODESolution ode_solve(const odefunc& f, const odejacobian& J, double t0, double tf, const arma::vec& U0) = 0;
    };

    class StepIVP {
        protected:
        double _step;
        double _next_t(double& k, double t, double tf) {
            double tt = t + k;
            if (tt > tf) {
                tt = tf;
                k = tf - t;
            }
            return tt;
        }
        void _check_step(double t0, double tf) {
            if (_step > tf - t0) throw std::runtime_error("step size (=" + std::to_string(_step) + ") exceeds domain range tf (=" + std::to_string(tf) + ") - t0 (=" + std::to_string(t0) + ") = " + std::to_string(tf-t0));
        }

        public:
        const double& step;
        explicit StepIVP(double step_size) : step(_step) {
            if (step_size <= 0) {
                throw std::invalid_argument("step (=" + std::to_string(step_size) + ") must be positive");
            }
            _step = step_size;
        }
    };

    class rk45 : public InitialValueProblem, public AdaptiveIVP {
        public:
        explicit rk45(double tol=1e-4, double minstep=1e-6) : AdaptiveIVP(tol,minstep) {}
        /* Dormand Prince adaptive runge kutta O(K^4) method for any explicit first order system of ODEs.
         * our equations are of the form u'(t) = f(t,u).
         * --- f  : f(t,u) [t must be the first variable, u the second].
         * --- t0, tf : t-range for solution.
         * --- U0  : initial value u(t0). */
        ODESolution ode_solve(const odefunc& f, double t0, double tf, const arma::vec& U0) override;
    };

    class rk45i : public InitialValueProblem, public AdaptiveIVP, public ImplicitIVP {
        public:
        explicit rk45i(double tol=1e-4, double minstep=1e-6) : AdaptiveIVP(tol,minstep) {}
        /* adaptive diagonally implicit runge kutta O(K^4) method for any explicit first order system of ODEs.
         * our equations are of the form u'(t) = f(t,u).
         * --- f  : f(t,u) [t must be the first variable, u the second].
         * --- t0, tf : t-range for solution.
         * --- U0  : initial value u(t0). */
        ODESolution ode_solve(const odefunc& f, double t0, double tf, const arma::vec& U0) override;
        /* adaptive diagonally implicit runge kutta O(K^4) method for any explicit first order system of ODEs.
         * our equations are of the form u'(t) = f(t,u).
         * --- f  : f(t,u) [t must be the first variable, u the second].
         * --- J : J(t,u) jacobian of f, i.e. df/du
         * --- t0, tf : t-range for solution.
         * --- U0  : initial value u(t0). */
        ODESolution ode_solve(const odefunc& f, const odejacobian& J, double t0, double tf, const arma::vec& U0) override;
    };

    class rk4 : public InitialValueProblem, public StepIVP {
        public:
        explicit rk4(double step_size=0.01) : StepIVP(step_size) {}
        /* runge kutta O(K^4) method for any explicit first order system of ODEs.
         * our equations are of the form u'(t) = f(t,u).
         * --- f  : f(t,u) [t must be the first variable, u the second].
         * --- t0, tf : t-range for solution.
         * --- U0  : initial value u(t0). */
        ODESolution ode_solve(const odefunc& f, double t0, double tf, const arma::vec& U0) override;
    };

    class rk5i : public InitialValueProblem, public StepIVP, public ImplicitIVP {
        public:
        explicit rk5i(double step_size = 0.01) : StepIVP(step_size) {}
        /* diagonally implicit runge kutta O(K^5) method for any explicit first order system of ODEs.
         * our equations are of the form u'(t) = f(t,u).
         * --- f  : f(t,u) [t must be the first variable, u the second].
         * --- t0, tf : t-range for solution.
         * --- U0  : initial value u(t0). */
        ODESolution ode_solve(const odefunc& f, double t0, double tf, const arma::vec& U0) override;
        /* diagonally implicit runge kutta O(K^5) method for any explicit first order system of ODEs.
         * our equations are of the form u'(t) = f(t,u).
         * --- f  : f(t,u) [t must be the first variable, u the second].
         * --- J : J(t,u) jacobian of f, i.e. df/du
         * --- t0, tf : t-range for solution.
         * --- U0  : initial value u(t0). */
        ODESolution ode_solve(const odefunc& f, const odejacobian& J, double t0, double tf, const arma::vec& U0) override;
    };

    class am1 : public InitialValueProblem, public StepIVP, public ImplicitIVP {
        public:
        explicit am1(double step_size = 0.01) : StepIVP(step_size) {}
        /* implicit Euler method O(k) for any explicit first order system of ODEs.
         * our equations are of the form u'(t) = f(t,u).
         * --- f  : f(t,u) [t must be the first variable, u the second].
         * --- t0, tf : t-range for solution.
         * --- U0  : initial value u(t0). */
        ODESolution ode_solve(const odefunc& f, double t0, double tf, const arma::vec& U0) override;
        /* implicit Euler method O(k) for any explicit first order system of ODEs.
         * our equations are of the form u'(t) = f(t,u).
         * --- f  : f(t,u) [t must be the first variable, u the second].
         * --- J : J(t,u) jacobian of f, i.e. df/du
         * --- t0, tf : t-range for solution.
         * --- U0  : initial value u(t0). */
        ODESolution ode_solve(const odefunc& f, const odejacobian& J, double t0, double tf, const arma::vec& U0) override;
    };

    class am2 : public InitialValueProblem, public StepIVP, public ImplicitIVP {
        public:
        explicit am2(double step_size = 0.01) : StepIVP(step_size) {}
        /* Adams-Multon O(k^2) method for any explicit first order system of ODEs.
         * our equations are of the form u'(t) = f(t,u).
         * --- f  : f(t,u) [t must be the first variable, u the second].
         * --- t0, tf : t-range for solution.
         * --- U0  : initial value u(t0). */
        ODESolution ode_solve(const odefunc& f, double t0, double tf, const arma::vec& U0) override;
        /* Adams-Multon O(k^2) method for any explicit first order system of ODEs.
         * our equations are of the form u'(t) = f(t,u).
         * --- f  : f(t,u) [t must be the first variable, u the second].
         * --- J  : jacobian of f, i.e. J = df/du
         * --- t0, tf : t-range for solution.
         * --- U0  : initial value u(t0). */
        ODESolution ode_solve(const odefunc& f, const odejacobian& J, double t0, double tf, const arma::vec& U0) override;
    };
    // --- BVPs ------------------- //
    class BoundaryValueProblem {
        protected:
        u_long _num_iter, _max_iter;
        double _tol;

        arma::vec _x;
        arma::mat _u;
        arma::mat _du;
        short _flag;

        u_long _check_dim(const arma::vec& x, const arma::mat& U) {
            u_long dim;
            if (U.n_rows == x.n_elem) dim = U.n_cols;
            else if (U.n_cols == x.n_elem) dim = U.n_rows;
            else throw std::runtime_error("dimension mismatch x.n_elem (=" + std::to_string(x.n_elem) + ") != any axis dim of U (" + std::to_string(U.n_rows) + ", " + std::to_string(U.n_cols) + ")");
            return dim;
        }

        void _check_x(const arma::vec& x) {
            if (not x.is_sorted()) throw std::runtime_error("require x to be sorted.");
        }

        public:
        const u_long& num_iter;
        const arma::vec& x;
        const arma::mat& u;
        const arma::mat& du;
        explicit BoundaryValueProblem(double tol, long max_iter) : num_iter(_num_iter), x(_x), u(_u), du(_du) {
            _tol = tol;
            _max_iter = max_iter;
            _flag = 0;
        }
        virtual void ode_solve(const odefunc& f, const boundary_conditions& bc, const arma::vec& x, const arma::mat& U) = 0;
        virtual void ode_solve(const odefunc& f, const odejacobian& J, const boundary_conditions& bc, const arma::vec& x, const arma::mat& U) = 0;
        virtual arma::mat operator()(const arma::vec& x) const = 0;

        std::string get_exit_flag() {
            std::string flag;
            if (_flag == 0) {
                flag = "solution successfully found over the specified domain.";
            } else if (_flag == 1) {
                flag = "solution could not be found within specified error tolerance.";
            } else if (_flag == 2) {
                flag = "NaN or infinite value encountered.";
            } else if (_flag == 3) {
                flag = "could not solve system of linear equations.";
            }
            return flag;
        }
    };

    class BVPk : public BoundaryValueProblem {
        protected:
        u_long k;
        std::vector<HSplineInterp> _sol;

        public:
        explicit BVPk(int order=4, double tol=1e-5, long max_iter=100) : BoundaryValueProblem(tol,max_iter) {
            if (order < 2) throw std::invalid_argument("require order (=" + std::to_string(order) + ") > 1");
            k = order;
        }
        /* solves general systems of  (potentially) nonlinear boudary value problems.
         * --- f  : u'(x) = f(x,u) is a vector valued function.
         * --- bc : bc(u[0], u[n-1]) == 0, boundary conditions function.
         * --- x  : x values, must be initialized (the solver will solve the bvp at these points), and must be sorted.
         * --- U  : u values, must be initialized, zeros should be fine, but an initial guess is ideal. */
        void ode_solve(const odefunc& f, const boundary_conditions& bc, const arma::vec& x, const arma::mat& U) override;
        /* solves general systems of  (potentially) nonlinear boudary value problems.
         * --- f  : u'(x) = f(x,u) is a vector valued function.
         * --- J  : jacobian of f, i.e. J = df/du
         * --- bc : bc(u[0], u[n-1]) == 0, boundary conditions function.
         * --- x  : x values, must be initialized (the solver will solve the bvp at these points), and must be sorted.
         * --- U  : u values, must be initialized, zeros should be fine, but an initial guess is ideal. */
        void ode_solve(const odefunc& f, const odejacobian& J, const boundary_conditions& bc, const arma::vec& x, const arma::mat& U) override;

        /* evaluate the solution at specified points in x. */
        arma::mat operator()(const arma::vec& x) const override;
    };
    
    class BVPCheb : public BoundaryValueProblem {
        protected:
        u_long num_pts;
        std::vector<Polynomial> _sol;

        public:
        explicit BVPCheb(long num_points = 32, double tol=1e-5, long max_iter=100) : BoundaryValueProblem(tol,max_iter) {
            if (num_points < 2) throw std::invalid_argument("require num_points (=" + std::to_string(num_points) + ") > 2");
            num_pts = num_points;
        }
        /* solves general systems of  (potentially) nonlinear boudary value problems.
         * --- f  : u'(x) = f(x,u) is a vector valued function.
         * --- bc : bc(u[0], u[n-1]) == 0, boundary conditions function.
         * --- x  : x values, must be initialized (the solver will solve the bvp at these points), and must be sorted.
         * --- U  : u values, must be initialized, zeros should be fine, but an initial guess is ideal. */
        void ode_solve(const odefunc& f, const boundary_conditions& bc, const arma::vec& x, const arma::mat& U) override;
        /* solves general systems of  (potentially) nonlinear boudary value problems.
         * --- f  : u'(x) = f(x,u) is a vector valued function.
         * --- J  : jacobian of f, i.e. J = df/du
         * --- bc : bc(u[0], u[n-1]) == 0, boundary conditions function.
         * --- x  : x values, must be initialized (the solver will solve the bvp at these points), and must be sorted.
         * --- U  : u values, must be initialized, zeros should be fine, but an initial guess is ideal. */
        void ode_solve(const odefunc& f, const odejacobian& jacobian, const boundary_conditions& bc, const arma::vec& x, const arma::mat& U) override;

        /* evaluate the solution at specified points in x. */
        arma::mat operator()(const arma::vec& x) const override;
    };

    class BVP3a : public BoundaryValueProblem {
        protected:
        std::vector<HSplineInterp> _sol;

        public:
        explicit BVP3a(double tol=1e-3, long max_iter=100) : BoundaryValueProblem(tol,max_iter) {}
        /* solves general systems of  (potentially) nonlinear boudary value problems.
         * --- f  : u'(x) = f(x,u) is a vector valued function.
         * --- bc : bc(u[0], u[n-1]) == 0, boundary conditions function.
         * --- x  : x values, must be initialized (the solver will solve the bvp at these points), and must be sorted.
         * --- U  : u values, must be initialized, zeros should be fine, but an initial guess is ideal. */
        void ode_solve(const odefunc& f, const boundary_conditions& bc, const arma::vec& x, const arma::mat& U) override;
        /* solves general systems of  (potentially) nonlinear boudary value problems.
         * --- f  : u'(x) = f(x,u) is a vector valued function.
         * --- J  : jacobian of f, i.e. J = df/du
         * --- bc : bc(u[0], u[n-1]) == 0, boundary conditions function.
         * --- x  : x values, must be initialized (the solver will solve the bvp at these points), and must be sorted.
         * --- U  : u values, must be initialized, zeros should be fine, but an initial guess is ideal. */
        void ode_solve(const odefunc& f, const odejacobian& J, const boundary_conditions& bc, const arma::vec& x, const arma::mat& U) override;

        arma::mat operator()(const arma::vec& x) const override;
    };

    // --- PDEs -------------------- //
    void poisson_helmholtz_2d(arma::mat& X, arma::mat& Y, arma::mat& U,
                const std::function<arma::mat(const arma::mat&, const arma::mat&)>& f,
                const std::function<arma::mat(const arma::mat&, const arma::mat&)>& bc,
                double k = 0,
                int m = 32);
}

#endif