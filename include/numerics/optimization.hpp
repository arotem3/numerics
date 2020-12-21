#ifndef NUMERICS_OPTIMIZATION_HPP
#define NUMERICS_OPTIMIZATION_HPP

namespace optimization {
    class VerboseTracker {
        protected:
        u_long max_iter;
        
        public:
        VerboseTracker(u_long m) {
            max_iter = m;
        }
        void header(const std::string& name="loss") {
            std::cout << "|" << std::right << std::setw(6) << std::setfill(' ') << "iter"
                    << "|" << std::right << std::setw(20) << std::setfill(' ') << "progress"
                    << "|" << std::right << std::setw(12) << std::setfill(' ') << name
                    << "|\n";
        }

        void iter(u_long iter, double fval) {
            std::string bar;
            float p = (float)iter/max_iter;
            for (int i=0; i < 20*p-1; ++i) bar += "=";
            bar += ">";
            std::cout << "|" << std::right << std::setw(6) << std::setfill(' ') << iter
                    << "|" << std::left << std::setw(20) << std::setfill(' ') << bar
                    << "|" << std::scientific << std::setprecision(4) << std::right << std::setw(12) << std::setfill(' ') << fval
                    << "|\r" << std::flush;
        }

        void success_flag() {
            std::cout << std::endl << "---converged to solution within tolerance---\n";
        }

        void max_iter_flag() {
            std::cout << std::endl << "---maximum number of iterations reached---\n";
        }

        void nan_flag() {
            std::cout << std::endl << "---NaN of Infinite value encountered---\n";
        }

        void empty_flag() {
            std::cout << std::endl;
        }
    };

    typedef std::function<arma::vec(const arma::vec&)> VecFunc;
    typedef std::function<arma::mat(const arma::vec&)> MatFunc;
    typedef std::function<double(const arma::vec&)> dFunc;
    //--- linear ---//

    /* pcg(x,A,b,[M,(M1,M2)],tol,max_iter) : solves the linear system Ax = b using the preconditioned conjugate gradient method.
     * --- x : initial guess and solution stored here
     * --- A : sparse or dense square sympd matrix or function computing A*x
     * --- b : as in A*x = b
     * --- M, M1, M2 : precondition matrix, M = M1*M2, or function which computes M(x) = inv(M)*x = inv(M2)*inv(M1)*x
     * --- tol : stopping criteria for measuring convergence to solution; iteration stops when ||Ax - b||/||b|| < tol
     * --- max_iter : maximum number of iterations after which the solver will stop regardless of convergence. When max_iter <= 0 ==> max_iter = x.n_elem */
    bool pcg(arma::vec& x, const arma::mat& A,    const arma::vec& b,                                                 double tol=1e-3, int max_iter=0);
    bool pcg(arma::vec& x, const arma::mat& A,    const arma::vec& b, const VecFunc& M,                               double tol=1e-3, int max_iter=0);
    bool pcg(arma::vec& x, const arma::mat& A,    const arma::vec& b, const arma::sp_mat& M,                          double tol=1e-3, int max_iter=0);
    bool pcg(arma::vec& x, const arma::mat& A,    const arma::vec& b, const arma::sp_mat& M1, const arma::sp_mat& M2, double tol=1e-3, int max_iter=0);

    bool pcg(arma::vec& x, const arma::sp_mat& A, const arma::vec& b,                                                 double tol=1e-3, int max_iter=0);
    bool pcg(arma::vec& x, const arma::sp_mat& A, const arma::vec& b, const VecFunc& M,                               double tol=1e-3, int max_iter=0);
    bool pcg(arma::vec& x, const arma::sp_mat& A, const arma::vec& b, const arma::sp_mat& M,                          double tol=1e-3, int max_iter=0);
    bool pcg(arma::vec& x, const arma::sp_mat& A, const arma::vec& b, const arma::sp_mat& M1, const arma::sp_mat& M2, double tol=1e-3, int max_iter=0);

    bool pcg(arma::vec& x, const VecFunc& A,      const arma::vec& b,                                                 double tol=1e-3, int max_iter=0);
    bool pcg(arma::vec& x, const VecFunc& A,      const arma::vec& b, const VecFunc& M,                               double tol=1e-3, int max_iter=0);
    bool pcg(arma::vec& x, const VecFunc& A,      const arma::vec& b, const arma::sp_mat& M,                          double tol=1e-3, int max_iter=0);
    bool pcg(arma::vec& x, const VecFunc& A,      const arma::vec& b, const arma::sp_mat& M1, const arma::sp_mat& M2, double tol=1e-3, int max_iter=0);

    /* gmres(x, A, b, [M, (M1,M2)], tol, max_iter) : solves the linear system Ax = b using the preconditioned Generalized Minimum RESidual method.
     * --- x : initial guess, solution is set to this variable
     * --- A : sparse or dense square matrix, or function which computes A(x) = A*x
     * --- b : as in Ax = b
     * --- M, M1, M2 : preconditioning matrix, M = M1*M2, or function which computes M(x) = inv(M)*x = inv(M2)*inv(M1)*x
     * --- tol : stopping criteria for convergence to solution; iteration stops when ||Ax-b||/||b|| < tol
     * --- max_iter : maximum number of iterations after which the solver will stop regardless of convergence. When max_iter <= 0 ==> max_iter = x.n_elem */
    bool gmres(arma::vec& x, const arma::mat& A,    const arma::vec& b,                                                 double tol=1e-3, int restart=0, int maxit=0);
    bool gmres(arma::vec& x, const arma::mat& A,    const arma::vec& b, const VecFunc& M,                               double tol=1e-3, int restart=0, int maxit=0);
    bool gmres(arma::vec& x, const arma::mat& A,    const arma::vec& b, const arma::sp_mat& M,                          double tol=1e-3, int restart=0, int maxit=0);
    bool gmres(arma::vec& x, const arma::mat& A,    const arma::vec& b, const arma::sp_mat& M1, const arma::sp_mat& M2, double tol=1e-3, int restart=0, int maxit=0);
    
    bool gmres(arma::vec& x, const arma::sp_mat& A, const arma::vec& b,                                                 double tol=1e-3, int restart=0, int maxit=0);
    bool gmres(arma::vec& x, const arma::sp_mat& A, const arma::vec& b, const VecFunc& M,                               double tol=1e-3, int restart=0, int maxit=0);
    bool gmres(arma::vec& x, const arma::sp_mat& A, const arma::vec& b, const arma::sp_mat& M,                          double tol=1e-3, int restart=0, int maxit=0);
    bool gmres(arma::vec& x, const arma::sp_mat& A, const arma::vec& b, const arma::sp_mat& M1, const arma::sp_mat& M2, double tol=1e-3, int restart=0, int maxit=0);
    
    bool gmres(arma::vec& x, const VecFunc& A,      const arma::vec& b,                                                 double tol=1e-3, int restart=0, int maxit=0);
    bool gmres(arma::vec& x, const VecFunc& A,      const arma::vec& b, const VecFunc& M,                               double tol=1e-3, int restart=0, int maxit=0);
    bool gmres(arma::vec& x, const VecFunc& A,      const arma::vec& b, const arma::sp_mat& M,                          double tol=1e-3, int restart=0, int maxit=0);
    bool gmres(arma::vec& x, const VecFunc& A,      const arma::vec& b, const arma::sp_mat& M1, const arma::sp_mat& M2, double tol=1e-3, int restart=0, int maxit=0);

    //--- nonlinear ---//
    class NonLinSolver {
        protected:
        u_long _max_iter;
        u_long _n_iter;
        short _exit_flag;
        double _xtol;
        double _ftol;
        bool _v;

        public:
        const u_long& n_iter;
        const short& exit_flag;
        
        void set_xtol(double xtol) {
            if (xtol <= 0) {
                throw std::invalid_argument("require xtol (=" + std::to_string(xtol) + ") > 0.");
            }
            _xtol = xtol;
        }

        void set_ftol(double ftol) {
            if (ftol <= 0) {
                throw std::invalid_argument("require ftol (=" + std::to_string(ftol) + ") > 0.");
            }
            _ftol = ftol;
        }

        void set_max_iter(long m) {
            if (m < 1) {
                throw std::invalid_argument("require max_iter (=" + std::to_string(m) + ") >= 1.");
            }
            _max_iter = m;
        }
        
        std::string get_exit_flag() const {
            std::string flag = "after " + std::to_string(_n_iter) + " iterations, ";
            if (_exit_flag == 0) flag += "first order conditions satisfied within ftol.";
            else if (_exit_flag == 1) flag += "solution could not be improved (step size < xtol).";
            else if (_exit_flag == 2) flag = "maximum number of iterations reached.";
            else if (_exit_flag == 3) flag += "NaN or infinite value encountered after ";
            else flag = "solver never called.";
            return flag;
        }

        explicit NonLinSolver(double xtol, double ftol, long maxit, bool verbose) : n_iter(_n_iter), exit_flag(_exit_flag) {
            set_xtol(xtol);
            set_ftol(ftol);
            set_max_iter(maxit);
            _n_iter = 0;
            _exit_flag = -1;
            _v = verbose;
        }
    };

    class QausiNewton : public NonLinSolver {
        protected:
        arma::vec _F;
        arma::mat _J;

        virtual void _initialize(const arma::vec& x, const VecFunc& f, const MatFunc* jacobian);
        virtual bool _step(arma::vec& dx, arma::vec& F1, const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) = 0;
        void _solve(arma::vec& x, const VecFunc& f, const MatFunc* jacobian);

        public:
        const arma::vec& fval;
        const arma::mat& Jacobian;

        explicit QausiNewton(double xtol, double ftol, long maxit, bool v=false) : NonLinSolver(xtol,ftol,maxit,v), fval(_F), Jacobian(_J) {}

        void fsolve(arma::vec& x, const VecFunc& f) {
            _solve(x, f, nullptr);
        }
        void fsolve(arma::vec& x, const VecFunc& f, const MatFunc& jacobian) {
            _solve(x, f, &jacobian);
        }
    };

    class Newton : public QausiNewton {
        protected:
        void _initialize(const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) override;
        bool _step(arma::vec& dx, arma::vec& F1, const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) override;

        public:
        explicit Newton(double xtol=1e-6, double ftol=1e-6, long maxiter=100, bool v=false) : QausiNewton(xtol, ftol, maxiter, v) {}
    };

    class Broyden : public QausiNewton {
        protected:
        bool _step(arma::vec& dx, arma::vec& F1, const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) override;

        public:
        /* initialize Broyden's method nonlinear solver specifying solver tolerance and maximum number of iterations. This object stores function values and jacobian for warm re-start of the solver. */
        explicit Broyden(double xtol=1e-6, double ftol=1e-6, long maxit=100, bool v=false) : QausiNewton(xtol, ftol, maxit, v) {}
    };

    class LmLSQR : public QausiNewton {
        protected:
        double _damping_param, _damping_scale, _lam;

        void _initialize(const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) override;
        bool _step(arma::vec& dx, arma::vec& F1, const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) override;

        public:
        /* initialize Levenberg-Marquardt solver initializing tol and max_iter. Solver stores jacobian for warm re-start of solver. */
        explicit LmLSQR(double xtol=1e-6, double ftol=1e-6, long maxit=100, bool v=false) : QausiNewton(xtol,ftol,maxit, v) {
            _damping_param = 1e-2;
            _damping_scale = 2.0;
        }

        /* specify the initial Levenberg-Marquardt damping parameter and scale:
            * --- tau : damping parameter, search direction (J'*J + tau*I)^{-1} * grad f
            * --- nu : scaling parameter, if we need to take a smaller step size we set tau = tau * nu */
        void set_damping_parameters(double tau, double nu) {
            if (tau <= 0) throw std::invalid_argument("require tau (=" + std::to_string(tau) + ") > 0");
            if (nu <= 1) throw std::invalid_argument("require nu (=" + std::to_string(nu) + ") > 1");
            _damping_param = tau;
            _damping_scale = nu;
        }
    };

    class TrustNewton : public QausiNewton {
        protected:
        double _delta;
        double _fh;

        void _initialize(const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) override;
        bool _step(arma::vec& dx, arma::vec& F1, const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) override;

        public:
        explicit TrustNewton(double xtol=1e-6, double ftol=1e-6, long maxit=100, bool v=false) : QausiNewton(xtol,ftol,maxit,v) {}
    };

    class MixFPI : public NonLinSolver {
        protected:
        u_int _steps_to_remember;

        public:
        explicit MixFPI(int steps_to_remember=5, double t=1e-3, long m=100, bool v=false) : NonLinSolver(t,t,m,v) {
            if (steps_to_remember < 1) {
                throw std::invalid_argument("require number of steps (=" + std::to_string(steps_to_remember) + ") >= 1");
            }
            _steps_to_remember = steps_to_remember;
        }

        /* Anderson mixing fixed point iteration. Finds solutions of the problem x = f(x).
         * --- x : initial guess and solution output.
         * --- f : vector function of x = f(x). */
        void fix(arma::vec& x, const VecFunc& f);
    };

    //--- univariate ---//

    /* adaptively deploys 3 different methods to find root of nonlinear equation
     * --- f  : function to find root of.
     * --- a,b : bracket for root.
     * --- tol : approximate error and stopping criteria. */
    double fzero(const std::function<double(double)>& f, double a, double b, double tol = 1e-8);
    
    /* finds local root of single variable nonlinear functions using newton's method.
     * --- f  : function to find root of.
     * --- df : derivative of f.
     * --- x : point near the root.
     * --- err : approximate error and stopping criteria. */
    double newton_1d(const std::function<double(double)>& f, const std::function<double(double)>& df, double x, double tol = 1e-8);
    
    double newton_1d(const std::function<double(double)>& f, double x, double tol=1e-8);

    /* secant methods for finding roots of single variable functions.
     * for added efficiency we attempt to bracket the root with an auxilary point.
     * --- f  : function to find root of.
     * --- a,b : bracket for root.
     * --- tol : approximate bound on error and stopping criteria */
    double secant(const std::function<double(double)>& f, double a, double b, double tol = 1e-8);
    
    /* bisection method for finding roots of single variable functions
     * --- f  : function to find root of.
     * --- a,b : bracket for root.
     * --- tol : approximate error and stopping criteria. */
    double bisect(const std::function<double(double)>& f, double a, double b, double tol = 1e-8);
    // --- optimization ----------- //
    /* step size approximator for quasi-newton methods based on strong wolfe conditions using the one dimensional nelder mead method.
     * --- f : objective function.
     * --- grad_f : gradient function.
     * --- x : current guess.
     * --- p : search direction.
     * --- c1 : wolfe constant 1.
     * --- c2 : wolfe constant 2. */
    double wolfe_step(const dFunc& f, const VecFunc& grad_f, const arma::vec& x, const arma::vec& p, double c1=1e-4, double c2=0.9);

    class GradientOptimizer : public NonLinSolver {
        protected:
        arma::vec _g;

        virtual void _initialize(const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian);
        virtual bool _step(arma::vec& dx, arma::vec& g1, const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) = 0;
        void _solve(arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian);

        public:
        const arma::vec& grad;
        explicit GradientOptimizer(double xtol, double ftol, long maxit, bool v) : NonLinSolver(xtol,ftol,maxit,v), grad(_g) {}

        void minimize(arma::vec& x, const dFunc& f, const VecFunc& grad_f) {
            _solve(x, f, grad_f, nullptr);
        }

        void minimize(arma::vec& x, const dFunc& f, const VecFunc& grad_f, const MatFunc& hessian) {
            _solve(x, f, grad_f, &hessian);
        }
    };

    class NewtonMin : public GradientOptimizer {
        protected:
        bool _step(arma::vec& dx, arma::vec& g1, const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) override;

        public:
        explicit NewtonMin(double xtol=1e-6, double ftol=1e-6, long maxit=100, bool verbose=false) : GradientOptimizer(xtol,ftol,maxit,verbose) {}
    };

    class BFGS : public GradientOptimizer {
        protected:
        double _wolfe_c1, _wolfe_c2;
        bool _use_fd;
        arma::mat _H;

        void _initialize(const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) override;
        bool _step(arma::vec& dx, arma::vec& g1, const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) override;

        public:
        const arma::mat& inv_hessian;
        explicit BFGS(double xtol=1e-6, double ftol=1e-6, long maxit=100, bool verbose=false, double wolfe1=1e-4, double wolfe2=0.9) : GradientOptimizer(xtol,ftol,maxit,verbose), inv_hessian(_H) {
            _wolfe_c1 = wolfe1;
            _wolfe_c2 = wolfe2;
            _use_fd = false;
        }

        void enable_finite_differences() {
            _use_fd = true;
        }

        void disable_finite_differences() {
            _use_fd = false;
        }
    };

    class LBFGS : public GradientOptimizer {
        protected:
        u_long _steps_to_remember;
        double _wolfe_c1, _wolfe_c2;
        numerics::CycleQueue<arma::vec> _S;
        numerics::CycleQueue<arma::vec> _Y;
        double _hdiag;

        void _lbfgs_update(arma::vec& p);
        void _initialize(const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) override;
        bool _step(arma::vec& dx, arma::vec& g1, const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) override;


        public:
        explicit LBFGS(long steps=5, double xtol=1e-6, double ftol=1e-6, long maxit=100, bool verbose=false, double wolfe1=1e-4, double wolfe2=0.9) : GradientOptimizer(xtol,ftol,maxit,verbose), _S(steps), _Y(steps) {
            _steps_to_remember = steps;
            _wolfe_c1 = wolfe1;
            _wolfe_c2 = wolfe2;
        }
    };

    class TrustMin : public GradientOptimizer {
        protected:
        double _delta;
        double _f0, _fh;

        void _initialize(const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) override;
        bool _step(arma::vec& dx, arma::vec& g1, const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) override;

        public:
        explicit TrustMin(double xtol=1e-6, double ftol=1e-6, long maxit=100, bool verbose=false) : GradientOptimizer(xtol,ftol,maxit,verbose) {}
    };

    class MomentumGD : public GradientOptimizer {
        protected:
        arma::vec _y;
        bool _line_min;
        double _damping_param, _alpha;

        bool _step(arma::vec& dx, arma::vec& g1, const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) override;
        
        public:
        const double& damping_parameter;
        const double& step_size;

        /* initializes gradient descent object. By default momentum is disabled, and line-minimization is used. To enable momentum, call set_step_size and specify a constant step size. */
        explicit MomentumGD(double xtol=1e-3, double ftol=1e-3, long maxit=1000, bool verbose=false) : GradientOptimizer(xtol,ftol,maxit,verbose), damping_parameter(_damping_param), step_size(_alpha) {
            _line_min = true;
            _damping_param = 0;
            _alpha = 0;
        }

        /* set the step size and momentum damping parameter.
            * --- alpha : (> 0) step size for gradient descent search.
            * --- damping_p : in [0,1) the momentum  */
        void set_step_size(double alpha, double damping_p=0.90) {
            if (alpha <= 0) {
                throw std::invalid_argument("require step size (=" + std::to_string(alpha) + ") > 0");
            }
            if ((damping_p < 0) or (damping_p >= 1)) {
                throw std::invalid_argument("require 0 <= damping parameter (=" + std::to_string(damping_p) + ") < 1");
            }
            _alpha = alpha;
            _damping_param = damping_p;
            _line_min = false;
        }
    };

    //--- linear contrained ---//
    /* simplex method for constrained linear minimization:
     *      min_x dot(f,x), s.t. conRHS * x < conLHS
     * --- x  : where solution is stored.
     * --- f  : z(x) = dot(f,x); function to maximize.
     * --- conRHS : right hand side of constraint equations.
     * --- conLHS : left hand side of constraint equations. */
    double simplex(arma::vec& x, const arma::vec& f, const arma::mat& conRHS, const arma::vec& conLHS);

    //--- gradient free ---//
    class NelderMead : public NonLinSolver {
        protected:
        double _step;
        double _expand;
        double _contract;
        double _shrink;
        double _side_length;

        arma::mat _init_simplex(const arma::vec& x);

        public:
        explicit NelderMead(double xtol=1e-3, double ftol=1e-3, long maxiter=1000, bool v=false) : NonLinSolver(xtol,ftol,maxiter,v) {
            _step = 1;
            _expand = 2;
            _contract = 0.5;
            _shrink = 0.5;
            _side_length = 1;
        }

        void set_step_size(double s) {
            if (s <= 0) throw std::invalid_argument("require step size (=" + std::to_string(s) + ") > 0");
            _step = s;
        }

        void set_expansion_param(double e) {
            if (e <= 1) throw std::invalid_argument("require expansion parameter (=" + std::to_string(e) + ") > 1");
            _expand = e;
        }

        void set_contraction_param(double c) {
            if ((c <= 0) or (c >= 1)) throw std::invalid_argument("require 0 < contraction parameter (=" + std::to_string(c) + ") < 1");
            _contract = c;
        }

        void set_shrinking_param(double s) {
            if ((s <= 0) or (s >= 1)) throw std::invalid_argument("require 0 < shrinking parameter (=" + std::to_string(s) + ") < 1");
            _shrink = s;
        }

        void set_initial_simplex_size(double s) {
            if (s <= 0) throw std::invalid_argument("require simplex size (=" + std::to_string(s) + ") > 0");
            _side_length = s;
        }

        /* minimize a multivariate function using the Nelder-Mead algorithm.
         * --- x : initial guess of minimum point, solution point will be assigned here.
         * --- f : f(x) function to minimize. */
        void minimize(arma::vec& x, const dFunc& f);
    };
    
    //--- univariate ---//
    /* fminbnd(f,a,b) : implementation of Brent's method for local minimization of a univariate function bounded on the interval (a,b) with f(a) and f(b) not necessarily defined. This method relys on the superlinear convergence rates of successive parabolic interpolation and the reliability of golden section search to find a minimum within O(log2(tol^-1)) iterations; where tol~O(1e-8)~O(sqrt(machine epsilon)).
     * --- f : objective function.
     * --- a : lower bound of interval.
     * --- b : upper bound of interval. */
    double fminbnd(const std::function<double(double)>& f, double a, double b, double tol=1e-8);

    /* fminsearch(f, x0, alpha=0) : implementation of the Nelder-Mead method in one dimension for unconstrained minimization.
     * --- f : objective function.
     * --- x0 : initial guess.
     * --- alpha : initial step size, if one isn't provided alpha will be set to 2*tol where tol~O(1e-8). */
    double fminsearch(const std::function<double(double)>& f, double x0, double alpha=0);
}

#endif