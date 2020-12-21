#ifndef NUMERICS_ODE_HPP
#define NUMERICS_ODE_HPP

namespace ode {
    typedef std::function<arma::vec(double,const arma::vec&)> odefunc;
    typedef std::function<arma::mat(double,const arma::vec&)> odejacobian;
    typedef std::function<arma::vec(const arma::vec&, const arma::vec&)> boundary_conditions;

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
        double _event_tol;
        long _stopping_event;
        std::vector<bool> _inc;
        std::vector<bool> _dec;
        std::vector<std::function<double(double, const arma::vec&)>> events;

        CycleQueue<arma::vec> _prev_u;
        CycleQueue<arma::vec> _prev_f;
        CycleQueue<double> _prev_t;
        std::unique_ptr<optimization::QausiNewton> solver;
        
        u_long _solver_miter;
        double _solver_xtol, _solver_ftol;

        std::vector<double> _t;
        std::vector<arma::vec> _U;

        double _event_handle(double t1, const arma::vec& u1);
        
        void _check_range(double t0, arma::vec& T);

        virtual double _initial_step_size() = 0;
        virtual double _step(double k, double& t1, arma::vec& u1, arma::vec& f1, const odefunc& f, const odejacobian* jacobian) = 0;
        void _update_solution(double& t1, arma::vec& u1, arma::vec& f1, bool full, bool is_grid_val);
        
        void _solve(const odefunc& f, const odejacobian* J, double t0, arma::vec T, const arma::vec& U0, bool full);

        public:
        const long& stopping_event;
        const std::vector<double>& t;
        const std::vector<arma::vec>& U;

        explicit InitialValueProblem(double event_tol, int n_step);

        void add_stopping_event(const std::function<double(double,const arma::vec&)>& event, const std::string& dir = "all");

        void solve_ivp(const odefunc& f, double t0, double tf, const arma::vec& U0, bool full=true);
        void solve_ivp(const odefunc& f, double t0, arma::vec T, const arma::vec& U0, bool full=false);
        void solve_ivp(const odefunc& f, const odejacobian& J, double t0, double tf, const arma::vec& U0, bool full=true);
        void solve_ivp(const odefunc& f, const odejacobian& J, double t0, arma::vec T, const arma::vec& U0, bool full=false);

        void as_mat(arma::vec& tt, arma::mat& uu);
    };

    struct ivpOpts {
        double atol;
        double rtol;
        double minstep;
        double solver_xtol;
        double solver_ftol;
        long solver_miter;
        double event_tol;
        double cstep;

        ivpOpts() {
            atol = 1e-6;
            rtol = 1e-3;
            minstep = 1e-6;
            solver_xtol = 1e-8;
            solver_ftol = 1e-8;
            solver_miter = 100;
            event_tol = 1e-4;
            cstep = 0.01;
        }
    };

    class AdaptiveIVP : public InitialValueProblem {
        protected:
        double _rtol;
        double _atol;
        double _min_step;

        double _initial_step_size() override {
            return std::min(100 * _min_step,1e-2);
        }

        virtual void _next_step_size(double& k, double res, double tol) = 0;

        public:
        explicit AdaptiveIVP(double rtol, double atol, double minstep, double event_tol, int n) : InitialValueProblem(event_tol, n) {
            if (minstep < 0) throw std::invalid_argument("minstep (=" + std::to_string(minstep) + ") must be non-negative.");
            if (rtol <= 0) throw std::invalid_argument("rtol (=" + std::to_string(rtol) + ") must be positive.");
            if (atol <= 0) throw std::invalid_argument("rtol (=" + std::to_string(rtol) + ") must be positive.");
            _min_step = minstep;
            _rtol = rtol;
            _atol = _atol;
        }
    };

    class CStepIVP : public InitialValueProblem {
        protected:
        double _cstep;

        double _initial_step_size() override {return _cstep;}

        public:
        explicit CStepIVP(double step, double event_tol, int n) : InitialValueProblem(event_tol, n) {
            if (step <= 0) {
                throw std::invalid_argument("step (=" + std::to_string(step) + ") must be positive");
            }
            _cstep = step;
        }
    };

    class rk45 : public AdaptiveIVP {
        protected:
        void _next_step_size(double& k, double res, double tol) override;

        double _step(double k, double& t1, arma::vec& u1, arma::vec& f1, const odefunc& f, const odejacobian* jacobian) override;
        
        public:
        /* Dormand Prince adaptive Runge-Kutta fourth order method.
        * see pg 178 of "Solving Ordinary Differential Equations I" by E. Hairer, and G. Wanner */
        explicit rk45(const ivpOpts& opts={}) : AdaptiveIVP(opts.rtol, opts.atol, opts.minstep, opts.event_tol, 1) {
            _solver_xtol = 1.0;
            _solver_ftol = 1.0;
            _solver_miter = 1;
        }
    };

    class rk34i : public AdaptiveIVP {
        protected:
        void _next_step_size(double& k, double res, double tol) override;
        arma::vec _v_solve(double tv, double k, const arma::vec& z, arma::vec& f1, const odefunc& f, const odejacobian* jacobian);

        double _step(double k, double& t1, arma::vec& u1, arma::vec& f1, const odefunc& f, const odejacobian* jacobian) override;
        
        public:
        /* Diagonally implicit adaptive fourth order method.
        * pg 107 in "Solving Ordinary Differential Equations II" by E. Hairer, G. Wanner */
        explicit rk34i(const ivpOpts& opts={}) : AdaptiveIVP(opts.rtol, opts.atol, opts.minstep, opts.event_tol, 1) {
            _solver_miter = opts.solver_miter;
            _solver_xtol = opts.solver_xtol;
            _solver_ftol = opts.solver_ftol;
        }
    };

    class rk4 : public CStepIVP {
        protected:
        const double rk4a[5] = {0.0, -567301805773.0/1357537059087.0, -2404267990393.0/2016746695238.0, -3550918686646.0/2091501179385.0, -1275806237668.0/842570457699.0};
        const double rk4b[5] = {1432997174477.0/9575080441755.0, 5161836677717.0/13612068292357.0, 1720146321549.0/2090206949498.0, 3134564353537.0/4481467310338.0, 2277821191437.0/14882151754819.0};
        const double rk4c[5] = {0.0, 1432997174477.0/9575080441755.0, 2526269341429.0/6820363962896.0, 2006345519317.0/3224310063776.0, 2802321613138.0/2924317926251.0};

        double _step(double k, double& t1, arma::vec& u1, arma::vec& f1, const odefunc& f, const odejacobian* jacobian) override;

        public:
        /* Five stage fourth order low storage rk4,
        * see "Fourth-order 2N-storage Runge-Kutta schemes" M.H. Carpenter and C. Kennedy */
        explicit rk4(const ivpOpts& opts={}) : CStepIVP(opts.cstep, opts.event_tol, 1) {
            _solver_xtol = 1.0;
            _solver_ftol = 1.0;
            _solver_miter = 1;
        }
    };

    class rk5i : public CStepIVP {
        protected:
        arma::vec _solve_v2(double k, double t, const arma::vec& z, const odefunc& f, const odejacobian* jacobian);
        arma::vec _solve_v3(double k, double t, const arma::vec& z, const odefunc& f, const odejacobian* jacobian);

        double _step(double k, double& t1, arma::vec& u1, arma::vec& f1, const odefunc& f, const odejacobian* jacobian) override;

        public:
        /* an L-stable diagonally implicit three stage fifth order method.
        * see "Numerical Methods for Ordinary Differential Equations" by J. C. Butcher */
        explicit rk5i(const ivpOpts& opts={}) : CStepIVP(opts.cstep, opts.event_tol, 1) {
            _solver_miter = opts.solver_miter;
            _solver_xtol = opts.solver_xtol;
            _solver_ftol = opts.solver_ftol;
        }
    };

    class am2 : public CStepIVP {
        protected:
        double _step(double k, double& t1, arma::vec& u1, arma::vec& f1, const odefunc& f, const odejacobian* jacobian) override;

        public:
        /* second order Adam-Moulton method (trapezoid rule). */
        explicit am2(const ivpOpts& opts={}) : CStepIVP(opts.cstep, opts.event_tol, 1) {
            _solver_miter = opts.solver_miter;
            _solver_xtol = opts.solver_xtol;
            _solver_ftol = opts.solver_ftol;
        }
    };

    // --- BVPs ------------------- //
    template<class SolutionT>
    class BoundaryValueProblem {
        protected:
        u_long _num_iter, _max_iter;
        double _tol;

        arma::vec _x;
        arma::mat _u;
        arma::mat _du;
        short _flag;

        std::vector<SolutionT> _sol;

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
        const std::vector<SolutionT>& solution;
        explicit BoundaryValueProblem(double tol, long max_iter) : num_iter(_num_iter), x(_x), u(_u), du(_du), solution(_sol) {
            _tol = tol;
            _max_iter = max_iter;
            _flag = 0;
        }
        virtual void solve_bvp(const odefunc& f, const boundary_conditions& bc, const arma::vec& x, const arma::mat& U) = 0;
        virtual void solve_bvp(const odefunc& f, const odejacobian& J, const boundary_conditions& bc, const arma::vec& x, const arma::mat& U) = 0;
        arma::mat operator()(const arma::vec& x) const {
            arma::mat out(_sol.size(), x.n_elem);
            for (u_long i=0; i < _sol.size(); ++i) {
                out.row(i) = _sol.at(i)(x).as_row();
            }
            return out;
        }
        arma::vec operator()(double x) const {
            arma::vec t = {x};
            return (*this)(t);
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

    class BVPk : public BoundaryValueProblem<PieceWisePoly> {
        protected:
        u_long k;

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
        void solve_bvp(const odefunc& f, const boundary_conditions& bc, const arma::vec& x, const arma::mat& U) override;
        /* solves general systems of  (potentially) nonlinear boudary value problems.
         * --- f  : u'(x) = f(x,u) is a vector valued function.
         * --- J  : jacobian of f, i.e. J = df/du
         * --- bc : bc(u[0], u[n-1]) == 0, boundary conditions function.
         * --- x  : x values, must be initialized (the solver will solve the bvp at these points), and must be sorted.
         * --- U  : u values, must be initialized, zeros should be fine, but an initial guess is ideal. */
        void solve_bvp(const odefunc& f, const odejacobian& J, const boundary_conditions& bc, const arma::vec& x, const arma::mat& U) override;
    };
    
    class BVPCheb : public BoundaryValueProblem<Polynomial> {
        protected:
        u_long num_pts;

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
        void solve_bvp(const odefunc& f, const boundary_conditions& bc, const arma::vec& x, const arma::mat& U) override;
        /* solves general systems of  (potentially) nonlinear boudary value problems.
         * --- f  : u'(x) = f(x,u) is a vector valued function.
         * --- J  : jacobian of f, i.e. J = df/du
         * --- bc : bc(u[0], u[n-1]) == 0, boundary conditions function.
         * --- x  : x values, must be initialized (the solver will solve the bvp at these points), and must be sorted.
         * --- U  : u values, must be initialized, zeros should be fine, but an initial guess is ideal. */
        void solve_bvp(const odefunc& f, const odejacobian& jacobian, const boundary_conditions& bc, const arma::vec& x, const arma::mat& U) override;
    };

    class BVP3a : public BoundaryValueProblem<PieceWisePoly> {
        public:
        explicit BVP3a(double tol=1e-3, long max_iter=100) : BoundaryValueProblem(tol,max_iter) {}
        /* solves general systems of  (potentially) nonlinear boudary value problems.
         * --- f  : u'(x) = f(x,u) is a vector valued function.
         * --- bc : bc(u[0], u[n-1]) == 0, boundary conditions function.
         * --- x  : x values, must be initialized (the solver will solve the bvp at these points), and must be sorted.
         * --- U  : u values, must be initialized, zeros should be fine, but an initial guess is ideal. */
        void solve_bvp(const odefunc& f, const boundary_conditions& bc, const arma::vec& x, const arma::mat& U) override;
        /* solves general systems of  (potentially) nonlinear boudary value problems.
         * --- f  : u'(x) = f(x,u) is a vector valued function.
         * --- J  : jacobian of f, i.e. J = df/du
         * --- bc : bc(u[0], u[n-1]) == 0, boundary conditions function.
         * --- x  : x values, must be initialized (the solver will solve the bvp at these points), and must be sorted.
         * --- U  : u values, must be initialized, zeros should be fine, but an initial guess is ideal. */
        void solve_bvp(const odefunc& f, const odejacobian& J, const boundary_conditions& bc, const arma::vec& x, const arma::mat& U) override;
    };

    // --- PDEs -------------------- //
    void poisson_helmholtz_2d(arma::mat& X, arma::mat& Y, arma::mat& U,
                const std::function<arma::mat(const arma::mat&, const arma::mat&)>& f,
                const std::function<arma::mat(const arma::mat&, const arma::mat&)>& bc,
                double k = 0,
                int m = 32);
}

#endif