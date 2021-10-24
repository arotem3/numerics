#ifndef NUMERICS_OPTIMIZATION_HPP
#define NUMERICS_OPTIMIZATION_HPP

namespace optimization {
    class Newton : public QausiNewton {
        protected:
        void _initialize(const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) override;
        bool _step(arma::vec& dx, arma::vec& F1, const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) override;

        public:
        explicit Newton(double xtol=1e-6, double ftol=1e-6, long maxiter=100, bool v=false) : QausiNewton(xtol, ftol, maxiter, v) {}
    };

    class Broyden : public QausiNewton {
        protected:
        void _initialize(const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) override;
        bool _step(arma::vec& dx, arma::vec& F1, const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) override;

        public:
        /* initialize Broyden's method nonlinear solver specifying solver
        tolerance and maximum number of iterations. This object stores function
        values and jacobian for warm re-start of the solver. The object Jacobian
        instead stores the current estimate of the inverse of the Jacobian. */
        explicit Broyden(double xtol=1e-6, double ftol=1e-6, long maxit=100, bool v=false) : QausiNewton(xtol, ftol, maxit, v) {}
    };

    class LmLSQR : public QausiNewton {
        protected:
        double _delta;

        void _initialize(const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) override;
        bool _step(arma::vec& dx, arma::vec& F1, const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) override;

        public:
        /* initialize Levenberg-Marquardt solver initializing tol and max_iter. Solver stores jacobian for warm re-start of solver. */
        explicit LmLSQR(double xtol=1e-6, double ftol=1e-6, long maxit=100, bool v=false) : QausiNewton(xtol,ftol,maxit, v) {}
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
    template<typename RealType, class Func>
    RealType fzero(const Func& f, RealType a, RealType b, RealType tol=1e-6) {
        const int max_iter = std::min<RealType>(std::pow(std::log2<RealType>((b-a)/tol)+1,2), 1.0e2); // will nearly never happen

        RealType c, d, e, fa, fb, fc, m=0, s=0, p=0, q=0, r=0, t, eps = std::numeric_limits<RealType>::epsilon();
        int k=0;
        fa = f(a); k++;
        fb = f(b); k++;
        if (std::abs(fa) == 0) return a;
        if (std::abs(fb) == 0) return b;

        if (fa*fb > 0) {
            throw std::invalid_argument("fzero() error: provided points do not bracket a simple root.");
        }
        
        c = a; fc = fa; d = b-a; e = d;

        while (true) {
            if (std::abs(fc) < std::abs(fb)) {
                a =  b;  b =  c;  c =  a;
                fa = fb; fb = fc; fc = fa;
            }
            m = (c-b)/2;
            t = 2*std::abs(b)*eps + tol;
            if (std::abs(m) < t || fb == 0) break; // convergence criteria
            if (k >= max_iter) {
                std::cerr << "fzero() error: could not converge within " << max_iter << " function evaluations (the estimated neccessary ammount).\n"
                        << "returing current best estimate.\n"
                        << "!!!---not necessarily a good estimate---!!!\n"
                        << "|dx| = " << std::abs(m) << " > " << tol << "\n";
                break;
            }

            if (std::abs(e) < t || std::abs(fa) < std::abs(fb)) { // bisection
                d = m; e = m;
            } else {
                s = fb/fa;
                if (a == c) { // secant
                    p = 2*m*s;
                    q = 1 - s;
                } else { // inverse quadratic
                    q = fa/fc;
                    r = fb/fc;
                    p = s*(2*m*q*(q-r)-(b-a)*(r-1));
                    q = (q-1)*(r-1)*(s-1);
                }

                if (p > 0) q = -q;
                else p = -p;

                s = e; e = d;

                if (2*p < 3*m*q - std::abs(t*q) && p < std::abs(0.5*s*q)) d = p/q;
                else {
                    d = m; e = m;
                }
            }
            a = b; fa = fb;

            if (std::abs(d) > t) b += d;
            else if (m > 0) b += t;
            else b -= t;

            fb = f(b); k++;

            if (fb*fc > 0) {
                c = a; fc = fa;
                e = b-a; d = e;
            }
        }
        return b;
    }
    
    /* finds local root of single variable nonlinear functions using newton's method.
     * --- f  : function to find root of.
     * --- df : derivative of f.
     * --- x : point near the root.
     * --- err : approximate error and stopping criteria. */
    template<typename RealType, class Func, class Deriv>
    RealType newton_1d(const Func& f, const Deriv& df, RealType x, RealType tol=1e-6) {
        if (tol <= 0) throw std::invalid_argument("newton_1d() error: error bound should be strictly positive, but tol=" + std::to_string(tol));
        constexpr int max_iter = 100;
        
        constexpr RealType eps = std::numeric_limits<RealType>::epsilon();
        u_long k = 0;
        RealType fx, fp, s;
        do {
            if (k >= max_iter) { // too many iterations
                std::cerr << "newton_1d() failed: too many iterations needed to converge." << std::endl
                        << "returing current best estimate."
                        << "!!!---not necessarily a good estimate---!!!" << std::endl
                        << "|f(x)| = " << std::abs(f(x)) << " > tolerance" << std::endl << std::endl;
                return x;
            }
            fx = f(x);
            fp = df(x);
            s = - fx / (fp + eps);
            x += s;
            k++;
        } while ((std::abs(fx) > tol) && (std::abs(s) > tol));
        return x;
    }
    
    template<typename RealType, class Func>
    RealType newton_1d(const Func& f, RealType x, RealType tol=1e-6) {
        auto df = [&](RealType u) -> RealType {
            static double eps = 2*std::numeric_limits<RealType>::epsilon();
            return deriv(f, u, u*eps, true, 2);
        };
        return newton_1d(f, df, x, tol);
    }
    
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