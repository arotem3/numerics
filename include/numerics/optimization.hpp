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
                    << "|\r";
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
    /* cgd(A,b,x,tol,max_iter) : solves the system Ax = b (or A'A*x = A'b) using conjugate gradient descent
     * --- A : system i.e. LHS (MAY BE OVERWITTEN IF 'A' IS NOT SYM POS DEF)
     * --- b : RHS (MAY BE OVERWRITTEN IF 'A' IS NOT SYM POS DEF)
     * --- x : initial guess and solution stored here
     * --- tol : stopping criteria for measuring convergence to solution.
     * --- max_iter : maximum number of iterations after which the solver will stop regardless of convergence */
    void cgd(arma::mat&, arma::mat&, arma::mat&, double tol = 1e-3, int max_iter = 0);

    /* cgd(A,b,x,tol,max_iter) : solves the sparse system Ax = b (or A'A*x = A'b) using conjugate gradient descent
     * --- A : sparse system i.e. LHS (MAY BE OVERWITTEN)
     * --- b : RHS (MAY BE OVERWRITTEN)
     * --- x : initial guess and solution stored here
     * --- tol : stopping criteria for measuring convergence to solution.
     * --- max_iter : maximum number of iterations after which the solver will stop regardless of convergence */
    void cgd(const arma::sp_mat&, const arma::mat&, arma::mat&, double tol = 1e-3, int max_iter = 0);

    //--- nonlinear ---//
    class NonLinSolver {
        protected:
        u_long _max_iter;
        u_long _n_iter;
        short _exit_flag;
        double _tol;
        bool _v;

        void _check_loop_parameters() {
            if ((_tol == 0) and (_max_iter == 0)) {
                throw std::runtime_error("tol and max_iter both set to zero which would result in an infinite loop.");
            }
        }

        public:
        const double& tol;
        const u_long& max_iter;
        const u_long& n_iter;
        const short& exit_flag;
        
        void set_tol(double t) {
            if (t < 0) {
                throw std::invalid_argument("require tol (=" + std::to_string(t) + ") > 0, or 0 -> use only max_iter");
            }
            _tol = t;
        }
        
        void set_max_iter(long m) {
            if (m < 0) {
                throw std::invalid_argument("require max_iter (=" + std::to_string(m) + ") >= 1, or 0 -> use only tol");
            }
            _max_iter = m;
        }
        
        std::string get_exit_flag() const {
            std::string flag;
            if (_exit_flag == 0) {
                flag = "converged to root within specified tolerance in ";
                flag += std::to_string(_n_iter) + " iterations.";
            } else if (_exit_flag == 1) {
                flag = "could not converge to the specified tolerance within ";
                flag += std::to_string(_max_iter) + " iterations.";
            } else if (_exit_flag == 2) {
                flag = "NaN or infinite value encountered after ";
                flag +=  std::to_string(_n_iter) + " iterations.";
            } else {
                flag = "solver never called.";
            }
            return flag;
        }
    
        explicit NonLinSolver(double t, long m, bool verbose) : tol(_tol), max_iter(_max_iter), n_iter(_n_iter), exit_flag(_exit_flag) {
            set_tol(t);
            set_max_iter(m);
            _n_iter = 0;
            _exit_flag = -1;
            _v = verbose;
        }
    };

    class Newton : public NonLinSolver {
        protected:
        bool _use_cgd;
        arma::vec _F;
        arma::mat _J;

        public:
        const arma::vec& fval;
        const arma::mat& Jacobian;
        void use_cgd() {
            _use_cgd = true;
        }
        void use_lu() {
            _use_cgd = false;
        }

        explicit Newton(double t=1e-3, long m=100, bool v=false) : NonLinSolver(t, m, v), fval(_F), Jacobian(_J) {
            _use_cgd = false;
        }

        /* finds a local root of a multivariate nonlinear system of equations using newton's method.
         * --- x : initial guess as to where the root, also where the root will be returned to.
         * --- f  : f(x) == 0, system of equations.
         * --- jacobian  : J(x) jacobian of system. */
        virtual void fsolve(arma::vec& x, const VecFunc& f, const MatFunc& jacobian);
    };

    class QausiNewton : public Newton {
        public:
        explicit QausiNewton(double t=1e-3, long m=100, bool v=false) : Newton(t, m, v) {}

        virtual void fsolve(arma::vec& x, const VecFunc& f) = 0;
    };

    class Broyd : public QausiNewton {
        public:
        /* initialize Broyden's method nonlinear solver specifying solver tolerance and maximum number of iterations. This object stores function values and jacobian for warm re-start of the solver. */
        explicit Broyd(double t=1e-3, long m=100, bool v=false) : QausiNewton(t, m, v) {}

        /* Broyden's method for local root finding of nonlinear system of equations, specify jacobian function as a warm start for the jacobian estimate
         * --- x : guess for root, also where root will be stored.
         * --- f : f(x) = 0 function for finding roots of.
         * --- jacobian : J(x) jacobian of f(x). */
        void fsolve(arma::vec& x, const VecFunc& f, const MatFunc& jacobian) override;
        
        /* Broyden's method for local root finding of nonlinear system of equations
         * --- f : f(x) = 0 function for finding roots of.
         * --- x : guess for root, also where root will be stored. */
        void fsolve(arma::vec& x, const VecFunc& f) override;

    };

    class LmLSQR : public QausiNewton {
        protected:
        double _damping_param, _damping_scale;

        public:
        /* initialize Levenberg-Marquardt solver initializing tol and max_iter. Solver stores jacobian for warm re-start of solver. */
        explicit LmLSQR(double t=1e-3, long m=100, bool v=false) : QausiNewton(t, m, v) {
            _damping_param = 1e-2;
            _damping_scale = 2;
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

        /* Levenberg-Marquardt damped least squares algorithm.
         * --- x : solution, initialized to a good guess.
         * --- f : f(x) == 0 function to find least squares solution of.
         * --- jacobian : J(x) jacobian function of f(x). */
        void fsolve(arma::vec& x, const VecFunc& f, const MatFunc& jacobian) override;

        /* Levenberg-Marquardt damped least squares algorithm.
         * --- f : function to find least squares solution of.
         * --- x : solution, initialized to a good guess. */
        void fsolve(arma::vec& x, const VecFunc& f) override;
    };

    class MixFPI : public NonLinSolver {
        protected:
        u_int _steps_to_remember;

        public:
        explicit MixFPI(int steps_to_remember=5, double t=1e-3, long m=100, bool v=false) : NonLinSolver(t,m,v) {
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
    double newton_1d(const std::function<double(double)>& f, const std::function<double(double)>& df, double x, double err = 1e-8);
    
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

        public:
        const arma::vec& grad;
        explicit GradientOptimizer(double t=1e-3, long m=100, bool v=false) : NonLinSolver(t,m,v), grad(_g) {}

        virtual void minimize(arma::vec& x, const dFunc& f, const VecFunc& grad_f) = 0;
    };

    class BFGS : public GradientOptimizer {
        protected:
        double _wolfe_c1, _wolfe_c2;
        bool _use_fd;
        bool _use_cgd;
        arma::mat _H;

        public:
        const arma::mat& inv_hessian;
        explicit BFGS(double t=1e-3, long m=100, bool v=false, double wolfe1=1e-4, double wolfe2=0.9) : GradientOptimizer(t,m,v), inv_hessian(_H) {
            _wolfe_c1 = wolfe1;
            _wolfe_c2 = wolfe2;
            _use_fd = false;
            _use_cgd = false;
        }

        void use_cgd() {
            _use_cgd = true;
        }

        void use_chol() {
            _use_cgd = false;
        }

        void enable_finite_differences() {
            _use_fd = true;
        }

        void disable_finite_differences() {
            _use_fd = false;
        }

        /* minimize(f, grad_f, x, max_iter) : Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm for local optimization
         * --- x : initial guess close to a local minimum, root will be stored here.
         * --- f : f(x) objective function
         * --- grad_f  : gradient of f(x). */
        void minimize(arma::vec& x, const dFunc& f, const VecFunc& grad_f) override;

        /* minimize(f, grad_f, hessian, x, max_iter) : Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm for local optimization
         * --- x : initial guess close to a local minimum, root will be stored here.
         * --- f : f(x) objective function
         * --- grad_f  : gradient of f(x).
         * --- hessian : hessian matrix of f(x). */
        void minimize(arma::vec& x, const dFunc& f, const VecFunc& grad_f, const MatFunc& hessian);
    };

    class LBFGS : public GradientOptimizer {
        protected:
        u_long _steps_to_remember;
        double _wolfe_c1, _wolfe_c2;
        CycleQueue<arma::vec> _S;
        CycleQueue<arma::vec> _Y;
        double _hdiag;
        void _lbfgs_update(arma::vec& p);

        public:
        explicit LBFGS(long steps=5, double t=1e-3, long m=100, bool v=false, double wolfe1=1e-4, double wolfe2=0.9) : GradientOptimizer(t,m,v), _S(steps), _Y(steps) {
            _steps_to_remember = steps;
            _wolfe_c1 = wolfe1;
            _wolfe_c2 = wolfe2;

        }
        /* Limited memory BFGS algorithm for local minimization
         * --- x : initial guess close to a local minimum, root will be stored here
         * --- f  : objective function to minimize
         * --- grad_f : gradient of objective function  */
        void minimize(arma::vec& x, const dFunc& f, const VecFunc& grad_f) override;
    };

    class MomentumGD :public GradientOptimizer {
        protected:
        bool _line_min;
        double _damping_param, _alpha;
        
        public:
        const double& damping_parameter;
        const double& step_size;

        /* initializes gradient descent object. By default momentum is disabled, and line-minimization is used. To enable momentum, call set_step_size and specify a constant step size. */
        explicit MomentumGD(double t=1e-3, long m=1000, bool v=false) : GradientOptimizer(t,m,v), damping_parameter(_damping_param), step_size(_alpha) {
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

        /* momentum gradient descent.
         * --- x : initial guess.
         * --- f : objective function
         * --- grad_f : gradient function. */
        void minimize(arma::vec& x, const dFunc& f, const VecFunc& grad_f) override;
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
        explicit NelderMead(double t=1e-3, long m=1000, bool v=false) : NonLinSolver(t,m,v) {
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

    //--- box contrained gradient free ---//
    class GeneticOptimizer : public NonLinSolver {
        protected:
        arma::vec _fitness(const std::function<double(const arma::vec&)>& f, const arma::mat& x, int n);
        arma::vec _diversity(arma::mat x);
        arma::rowvec _cross_over(const arma::rowvec& a, const arma::rowvec& b);

        double _reproduction_rate, _mutation_rate, _search_radius, _diversity_weight;
        u_long _population_size, _diversity_cutoff, _random_seed;

        public:
        explicit GeneticOptimizer(long seed=0, double t=1e-1, long m=100) : NonLinSolver(t,m,false) {
            _reproduction_rate = 0.5;
            _mutation_rate = 0.5;
            _diversity_weight = 0.2;
            _search_radius = 1;
            _population_size = 100;
            _diversity_cutoff = 30;
            _random_seed = seed;
        }

        void set_search_radius(double r) {
            if (r <= 0) throw std::invalid_argument("require search radius (=" + std::to_string(r) + ") > 0");
            _search_radius = r;
        }

        void set_population_size(long popsize) {
            if (popsize < 2) throw std::invalid_argument("require population size (=" + std::to_string(popsize) + ") > 1");
            _population_size = popsize;
        }

        void set_diversity_parameters(double reproduction=0.5, double mutation=0.5, double diversity=0.2, long cutoff=30) {
            if ((reproduction <= 0) or (reproduction >= 1)) throw std::invalid_argument("require 0 < reproduction rate (=" + std::to_string(reproduction) + ") < 1");
            _reproduction_rate = reproduction;
            if ((mutation <= 0) or (mutation >= 1)) throw std::invalid_argument("require 0 < mutation rate (=" + std::to_string(mutation) + ") < 1");
            _mutation_rate = mutation;
            if ((diversity <= 0) or (diversity >= 1)) throw std::invalid_argument("require 0 < diversity rate (=" + std::to_string(diversity) + ") < 1");
            _diversity_weight = diversity;
            if (cutoff < 0) throw std::invalid_argument("require diversity iteration cutoff (=" + std::to_string(cutoff) + ") >= 0");
            _diversity_cutoff = cutoff;
        }
    
        /* box constrained maximization using a genetic algorithm.
         * --- x  : vec where the maximum is.
         * --- f  : double = f(x) function to maximize.
         * --- lower_bound,upper_bound : bounds for box constraints. */
        void maximize(arma::vec& x, const dFunc& f, const arma::vec& lower_bound, const arma::vec& upper_bound);
        
        /* maximize(f,x) : unconcstrained maximization using a genetic algorithm.
         * --- x  : vec where the maximum is.
         * --- f  : double = f(x) function to maximize. */
        void maximize(arma::vec& x, const dFunc& f);
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