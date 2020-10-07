[//]: # (pandoc -s documentation.md -o documentation.pdf -V geometry:margin=0.75in --highlight-style tango)
# `numerics.hpp` Documentation

## Table of Contents
- [`numerics.hpp` Documentation](#numericshpp-documentation)
  - [Table of Contents](#table-of-contents)
  - [Utility Methods](#utility-methods)
    - [Modulus operation](#modulus-operation)
    - [Meshgrid](#meshgrid)
    - [Polynomial Derivatives and Integrals](#polynomial-derivatives-and-integrals)
    - [Sample Discrete Distributions](#sample-discrete-distributions)
  - [Quadrature and Finite Differences](#quadrature-and-finite-differences)
    - [Integration](#integration)
      - [Example](#example)
  - [Discrete Derivatives](#discrete-derivatives)
    - [Finite Differences](#finite-differences)
      - [Example](#example-1)
    - [Derivatives in Higher Dimension](#derivatives-in-higher-dimension)
      - [Example](#example-2)
    - [Spectral Derivatives](#spectral-derivatives)
      - [Example](#example-3)
  - [Linear Root Finding and Optimization](#linear-root-finding-and-optimization)
    - [Conjugate Gradient Method](#conjugate-gradient-method)
    - [Linear Programming](#linear-programming)
  - [Nonlinear Root Finding](#nonlinear-root-finding)
    - [Newton's method](#newtons-method)
    - [Quasi-Newton Solvers](#quasi-newton-solvers)
    - [Broyden's Method](#broydens-method)
    - [Levenberg-Marquardt Trust Region/Damped Least Squares](#levenberg-marquardt-trust-regiondamped-least-squares)
    - [Fixed Point Iteration with Anderson Mixing](#fixed-point-iteration-with-anderson-mixing)
    - [fzero](#fzero)
    - [Secant](#secant)
    - [Bisection method](#bisection-method)
  - [Nonlinear Optimization](#nonlinear-optimization)
    - [fminbnd](#fminbnd)
    - [fminsearch](#fminsearch)
    - [Multivariate Minimization](#multivariate-minimization)
    - [Broyden–Fletcher–Goldfarb–Shanno algorithm](#broydenfletchergoldfarbshanno-algorithm)
    - [Limited Memory BFGS](#limited-memory-bfgs)
    - [Momentum Gradient Descent](#momentum-gradient-descent)
    - [Nelder-Mead Gradient Free Minimization](#nelder-mead-gradient-free-minimization)
    - [Genetic Maximization Algorithm](#genetic-maximization-algorithm)
  - [Interpolation](#interpolation)
    - [Polynomials](#polynomials)
    - [Piecewise Polynomials](#piecewise-polynomials)
    - [Sinc interpolation](#sinc-interpolation)
  - [Namespace `neighbors`](#namespace-neighbors)
  - [Data Science & Machine Learning](#data-science--machine-learning)
    - [Feature Maps](#feature-maps)
      - [Cubic Kernel](#cubic-kernel)
      - [Polynomial Features](#polynomial-features)
    - [K-Fold Train Test Split](#k-fold-train-test-split)
      - [Example](#example-4)
    - [Label Encoding](#label-encoding)
    - [Data Binning](#data-binning)
      - [Example](#example-5)
    - [K-Means Clustering](#k-means-clustering)
    - [Linear Models](#linear-models)
    - [LASSO Regression](#lasso-regression)
    - [Ridge Regression](#ridge-regression)
    - [Splines](#splines)
    - [Logistic Regression](#logistic-regression)
    - [Neural Networks and SGD Models](#neural-networks-and-sgd-models)
    - [Linear SGD Models](#linear-sgd-models)
    - [Neural Network Models](#neural-network-models)
    - [k Nearest Neighbors Estimators](#k-nearest-neighbors-estimators)
    - [k Nearest Neighbors Classifier](#k-nearest-neighbors-classifier)
    - [k Nearest Neighbors Regressor](#k-nearest-neighbors-regressor)
    - [Kernel Estimators](#kernel-estimators)
    - [Kernel Density Estimation](#kernel-density-estimation)
    - [Kernel Smoothing](#kernel-smoothing)
- [`numerics::ode` Documentation](#numericsode-documentation)
  - [Table of Contents](#table-of-contents-1)
  - [Differentiation Operators](#differentiation-operators)
  - [Initial Value Problem Solvers](#initial-value-problem-solvers)
    - [Dormand-Prince 4/5](#dormand-prince-45)
    - [Runge-Kutta Fourth Order](#runge-kutta-fourth-order)
    - [Runge-Kutta Implicit Fourth Order](#runge-kutta-implicit-fourth-order)
    - [Runge-Kutta Implicit Fifth Order](#runge-kutta-implicit-fifth-order)
    - [Backwards Euler](#backwards-euler)
    - [Adams-Moulton Second Order](#adams-moulton-second-order)
    - [IVP Events](#ivp-events)
  - [Boundary Value Problems Solver](#boundary-value-problems-solver)
    - [Variable Order Method](#variable-order-method)
    - [Chebyshev Spectral Method](#chebyshev-spectral-method)
    - [Lobatto IIIa method](#lobatto-iiia-method)
    - [Poisson Solver](#poisson-solver)

all definitions are members of namespace `numerics`, all examples assume:
```cpp
using namespace std;
using namespace arma; // typically only vec and mat
using namespace numerics;
using namespace ode;
```

## Utility Methods

### Modulus operation
```cpp
int mod(int a, int b);
```
In `C++`, the modulo operator (`a%b`) returns the remainder from dividing `a` by `b`, while `mod(a,b)` function returns $a(\text{mod } b)$ which is distinct from `a%b` whenever `a` is negative. 

Example:

```cpp
int a = -3 % 11; // a = 2
int b = mod(-3, 11); // b = 8
```

### Meshgrid
```cpp
void meshgrid(arma::mat& xgrid, arma::mat& ygrid,
              const arma::vec& x, const arma::vec& y);
void meshgrid(arma::mat& xgrid, const arma::vec& x);
```
This function constructs a 2D grid of points (representing a region of the xy-plane), with non-uniform points defined in `x` and `y`.

Example:

```cpp
vec x = linspace(0, 2, 10);
vec y = -cos(regspace(-M_PI, M_PI, 20))
mat XX, YY;
meshgrid(XX, YY, x, y);
```

### Polynomial Derivatives and Integrals
Given a polynomial coefficient vector (produced from `arma::polyfit`), we can produce derivatives of the polynomial with the following function:
```cpp
arma::vec polyder(const arma::vec& p, unsigned int k = 1);
```
where `p` is the coefficient vector, and `k` is order of the derivative. By default, `k = 1` which corresponds to the first derivative (and `k = 2` is the second derivative, and so on). The output is also a polynomial coefficient vector.

We can integrate the polynomial:
```cpp
arma::vec polyint(const arma::vec& p, double c = 0);
```
where `p` is the coefficient vector, and `c` is the constant of integration; by default `c = 0`. The output is also a polynomial coefficient vector.

### Sample Discrete Distributions

Given a discrete probability mass function, we can produce a random sample.

```cpp
arma::uvec sample_from(int n, const arma::vec& pmf, const arma::uvec& labels=arma::uvec());

int sample_from(const arma::vec& pmf, const arma::uvec& labels=arma::uvec());
```
The output is either a set of integers between 0 and n-1 refering to the index associated with the pmf, or a set of labels sampled from `labels` using the pmf for random indexing.

## Quadrature and Finite Differences

### Integration
If we simply need a decent approximation of an integral of a function $f:[a,b]\to\R$, then `integrate()` is more than sufficient:
```cpp
double integrate(const function<double(double)>& f, double a, double b, const string& method="lobatto");
```
This function uses the default parameters for the integration `method` selected to approximate the integral. The `method` can be one of `"lobatto","simpson","chebyshev"`.

The `"lobatto"` method corresponds to an adaptive Gauss-Lobatto 4 point approximation with a 7 point error estimate. This method adaptively modifies the evaluation mesh to bound the error below the requested tolerance. The Lobatto scheme is ideal for smooth functions which are easy to evaluate because this scheme doess not recycle function evaluations. This method has the stand-alone function
```cpp
double lobatto_integral(const function<double(double)>& f, double a, double b, double tol = 1e-8);
``` 

The `"simpson"` method corresponds to an adaptive Simpson's method scheme which uses quadratic interpolation to approximate the integral over an interval. This method modifies the evaluation mesh by comparing the approximation on each element of the mesh with an approximation over two elements in the same range in order to bound the error bellow a specified tolerance. This scheme is generally applicable and recycles function evaluations. This method has a stand-alone function with the added benefit of storing function evaluations to a standard library `map`.
```cpp
double simpson_integral(map<double,double>& fmap, const function<double(double)>& f, double a, double b, double tol = 1e-5);

double simpson_integral(const function<double(double)>& f, double a, double b, double tol = 1e-5);
```

Finally, the `"chebyshev"` method corresponds to integration by polynomial interpolation. This scheme is only idea for very smooth functions, but can achieve very high accuracy with very few function evaluations relative to both `"simpson"` and `"lobatto"`. Traditionally, polynomial interpolation takes $\mathcal O(n^3)$ time, but since we can choose the set of points to interpolate over, we can use Chebyshev nodes and integrate the function in $\mathcal O(n\log n)$ time using the fast fourier transform (caveat, this implementation is not optimal, it is $\mathcal O(n^2)$ for now--still an improvement). The resulting error decays exponentially with $n$ (for functions with atleast $n-1$ continuous derivatives). This method has a stand-alone function,
```cpp
double chebyshev_integral(const function<double(double)>& f, double a, double b, uint m = 32);
```
Where `m` is the number of function evaluations which achieves high accuracy for modest values.

_**note:**_ If the function is not (atleast) continuous, the approximation may quickly become ill conditioned.

#### Example
```cpp
double f(double x) return exp(-x*x); // e^(-x^2)
double lower_bound = 0;
double upper_bound = 3.14;

double I = integrate(f, lower_bound, upper_bound);

double I_simp = integrate(f, lower_bound, upper_bound, 1e-6,"simpson");
```

## Discrete Derivatives
### Finite Differences
```cpp
double deriv(const function<double(double)>& f, double x,
            double h = 1e-2, bool catch_zero = true);
```
This function uses 4th order finite differences with step size `h` to approximate the derivative at a point `x`. We can also ask the derivative function to catch zeros, i.e. round to zero whenever `f'(x) < h`; this option can improve the numerical stability of methods relying on results from `deriv()`, e.g. approximating sparse Hessians or Jacobians.

**note:** `deriv()` may never actually evaluate the function at the point of interest, which is ideal if the function is not well behaved there.

#### Example
```cpp
double f(double x) return sin(x);

double x = 0;

double df = deriv(f,x); // should return ~1.0

double g(double x) return cos(x);

double dg = deriv(g,x,1e-2,false); // should return ~0.0 
/*
in this case we are better off if catch_zero = true because d(cos x)/dx = 0 for x = 0.
*/
```
### Derivatives in Higher Dimension
We can also approximate gradient vectors and Jacobian matrices:
```cpp
vec grad(const function<double(const vec&)> f,
            const vec& x,
            double h = 1e-2,
            bool catch_zeros = true);

mat approx_jacobian(const function<vec(const vec&)> f,
                     const vec& x,
                     double h = 1e-2, bool catch_zero = true);

vec jacobian_diag(const function<vec(const vec&)> f,
                    const vec& x,
                    double h = 1e-2);
```
These functions are wrappers for the `deriv()` function. So the functionality is similar. The function `jacobian_diag()` computes only the diagonal of the jacobian matrix, which may only make sense when the jacobian is a square matrix.

#### Example
```cpp
double f(const vec& x) return dot(x,x); // return x^2 + y^2 + ...

vec F(const vec& x) return x%(x+1); // returns x.*(x+1)

vec x = {0,1,2};

vec gradient = grad(f,x); // should return [2*0, 2*1, 2*2] = [0,2,4]

mat Jac = approx_jacobian(F,x);
/* Jac = [1.0   0.0   0.0
          0.0   3.0   0.0
          0.0   0.0   4.0];
*/
```

### Spectral Derivatives
Given function defined over an interval, we can approximate the derivative of the function with spectral accuracy using the `FFT`. The function `spectral_deriv()` does this using polynomial interpolation. We sample the function at chebyshev nodes scaled to the interval [a,b]; the function returns a polynomial object that can be evaluated anywhere on the interval. We can specify more sample points for more accuracy.
```cpp
Polynomial spectral_deriv(const function<double(double)>& f, double a, double b, uint sample_points = 50);
```
Further details of the `Polynomial` class can be found [here](#polynomials).

#### Example
```cpp
double f(double x) return sin(x*x);

auto df = spectral_deriv(f, -2, 2);

vec x = linspace(-2,2);
vec derivative = df(x);
```

For more discrete derivatives see [`numerics::ode`](#`numerics::ode`-documentation).

## Linear Root Finding and Optimization
The header file `optimization.hpp` declares the following types for convinience:
```cpp
typedef function<vec(const vec&)>       VecFunc;    // R^n -> R^k
typedef function<mat(const vec&)>       MatFunc;    // R^n -> R^{k x n}
typedef function<double(const vec&)>    dFunc;      // R^n -> R
```
### Conjugate Gradient Method
Armadillo features a very robust `solve()` and `spsolve()` direct solvers for linear systems, but in the case where less precise solutions of very large systems (especially sparse systems) iterative solvers may be more efficient. The functions `cgd()` solve systems of linear equations $A \mathbf{x}=\mathbf{b}$ when $A$ is symmetric positive definite (sparse or dense), or in the least squares sense $A^TA\mathbf{x}=A^T\mathbf{b}$ by conjugate gradient method. The righthand side $b$ can be either a single column vector or a matrix.
```cpp
void cgd(mat& x, const mat& A, const mat& b, double tol = 1e-3, int max_iter = 0);
void cgd(mat& x, const sp_mat& A, const mat& b, double tol = 1e-3, int max_iter = 0);
```
if `max_iter <= 0`, then `max_iter = b.n_rows`.

### Linear Programming

For solving linear __*maximization*__ problems with linear constraints, we have the simplex algorithm that computes solutions using partial row reduction of linear system of equations. The simplex method solves problems of the form:
$$\max_x f^T x$$
$$A x \preceq b$$
```cpp
double simplex(arma::vec& x, const arma::vec& f, const arma::mat& A, const arma::vec& b);
```
The function returns the __*maximum*__ value within a convex polygon defined by $A x\preceq b$. The argument of the maximum is stored in `x`.

## Nonlinear Root Finding
All of the nonlinear solver inherit from the `nlsolver` class:
```cpp
class NonLinSolver {
    public:
    const double& tol;
    const u_long& max_iter;
    const u_long& n_iter;
    const short& exit_flag;

    explicit NonLinSolver(double tol, long maxiter, bool verbose);
    
    void set_tol(double t);
    void set_max_iter(long m);
    std::string get_exit_flag() const;
};
```
All of the solvers have `tol` speicifications which varies in interpretation for each method, but it is essentially a small parameter that determines convergence. In general, `tol` is either
$$\|x_{k+1} - x_k\|_\infty < \texttt{tol},$$
or
$$\|f(x_{k+1})\|_\infty < \texttt{tol}.$$
The parameter `maxiter` lets you specify the maximum iterations to compute before stopping the scheme; this parameter also varies from scheme to scheme based on convergence properties and cost per iteration of each method.

If `verbose == true`, then the solver will display a progress bar showing the number of iterations complete relative to `max_iter`, and display a message upon completion.

The parameter `exit_flag` takes on values `{-1,0,1,2}` where:
* -1 : the solver was never given a problem to solve, e.g. `fsolve` was never called.
* 0 : the solver successfully converged.
* 1 : the maximum number of iterations was reached.
* 2 : a NaN or Infinite value was encountered during a function evaluation.

The function `get_exit_flag` will provide a string which explains why the solver stopped, i.e. one of the options above.

### Newton's method
This is an implementation of Newton's method for systems of nonlinear equations. A jacobian function of the system of equations is required. As well as a good initial guess:
```cpp
class Newton : public NonLinSolver {
    public:
    const vec& fval;
    const mat& Jacobian;

    explicit Newton(double tol=1e-3, long maxiter=100, bool verbose=false);

    void use_cgd();
    void use_lu();

    virtual void fsolve(vec& x, const VecFunc& f, const MatFunc& jacobian);
};
```
The function `fsolve` solves the system $f(x) = 0$. The initial guess should be stored in `x` and this value will be updated with the solution found by Newton's method. The functions `use_cgd`, and `use_lu` allows the user to specify how to invert the jacobian at each iteration where `use_cgd` toggles the use of conjugate gradient method, while `use_lu` toggles the use of `arma::solve()`; by default Newton's method inverts the jacobian directly.

The solver also stores the final value of $f(x)$ in `fval` for verifying the solution, as well as the final value of the jacobian in `Jacobian`.

There is also a single variable version:
```cpp
double newton_1d(const function<double(double)>& f,
              const function<double(double)>& df,
              double x,
              double err = 1e-5);
```
### Quasi-Newton Solvers
The following solvers inherit from the virtual class `QuasiNewton`:
```cpp
class QausiNewton : public Newton {
    public:
    explicit QausiNewton(double tol=1e-3, long maxiter=100, bool verbose=false);

    virtual void fsolve(arma::vec& x, const VecFunc& f) = 0;
};
```

### Broyden's Method
This solver is similar to Newton's method, but does not require the jacobian matrix to be evaluated at every step; instead, the solver takes rank 1 updates of the inverse of the estimated Jacobian using the secant equations [(wikipedia)](https://en.wikipedia.org/wiki/Broyden%27s_method). Providing a jacobian function does improve the scheme (especially at initialization), but this solver requires far fewer Jacobian evaluations than Newton's method. If none is provided the initial jacobian is computed using finite differencing as this drastically improves the convergence.
```cpp
class Broyd : public QausiNewton {
    public:
    explicit Broyd(double tol=1e-3, long maxiter=100, bool verbose=false);

    void fsolve(vec& x, const VecFunc& f, const MatFunc& jacobian) override;
    void fsolve(vec& x, const VecFunc& f) override;
};
```

### Levenberg-Marquardt Trust Region/Damped Least Squares
This solver performs Newton like iterations, replacing the Jacobian with a damped least squares version [(wikipedia)](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm). The jacobian is updated with Broyden's rank 1 updates of the Jacobian itself (rather than the inverse). Just like Broyden's method, providing a jacobian function is helpful but not necessary. When a jacobian is not provided, it will be approximated by finite-differences.
```cpp
class LmLSQR : public QausiNewton {
        public:
        explicit LmLSQR(double tol=1e-3, long maxiter=100, bool verbose=false);

        void set_damping_parameters(double tau, double nu);

        void fsolve(vec& x, const VecFunc& f, const MatFunc& jacobian) override;
        void fsolve(vec& x, const VecFunc& f) override;
    };
```
This solver has two tuneable parameters affecting convergence $\tau$ and $\nu$. The search direction is determined by $(J^T J + \tau I)^{-1} \nabla f$, the provided value of $\tau$ is an initial guess, as the algorithm searches for an optimal value as it proceeds. The default value is `1.0e-2`. The parameter $\nu$ is a part of the line search procedure for an optimal $\tau$, if a smaller step is necessary then we set
$$\tau_{k+1} = \nu \tau_k.$$ 
The default value is `2.0`. Both values must be positive and can be set with `set_damping_parameters`.

### Fixed Point Iteration with Anderson Mixing
Solves problems of the form $x = g(x)$. It is possible to solve a subclass of these problems using fixed point iteration i.e. $x_{n+1} = g(x_n)$, but more generally we can solve these systems using Anderson accelaration:
$$x_{n+1} = \sum_{i=p}^n c_i g(x_i),$$
where $1 \leq p \leq n$ and $\sum c_i = 0.$
```cpp
class MixFPI : public NonLinSolver {
    public:
    explicit MixFPI(
        int steps_to_remember=5,
        double tol=1e-3,
        long maxiter=100, 
        bool verbose=false
    );

    void fix(arma::vec& x, const VecFunc& f);
};
```
The parameter `steps_to_remember` is equal to $n-p$, i.e. the number of previous values to use in the updating strategy.

### fzero
Adaptively selects between secant method, and inverse interpolation to find a *simple* root of a single variable function in the interval `[a,b]`.
```cpp
double fzero(const function<double(double)>& f, double a, double b, double tol=1e-8);
```

### Secant
Uses the secant as the approximation for the derivative used in Newton's method. Attempts to bracket solution for faster convergence, so providing an interval rather than two initial guesses is best.
```cpp
double secant(const function<double(double)>& f, double a, double b, double tol=1e-8);
```

### Bisection method
Uses the bisection method to find the solution to a nonlinear equation within an interval.
```cpp
double bisect(const function<double(double)>& f, double a, double b, double tol=1e-8);
```

## Nonlinear Optimization

### fminbnd
provided a continuous function $f:(a,b)\rightarrow\mathbb{R}$ which is not necessarily continuous at the end points, we can find a local minimum of $f$ within a small number of steps (the number of function evaluations bounded by $\approx 2.88[\log_2 \frac{b-a}{\epsilon}]^2\approx 100$ function evaluations, when we select $\texttt{tol}=\epsilon=10^{-8}\times(b-a)$). The method:
```cpp
double fminbnd(const function<double(double)>& f, double a, double b, double tol=1e-8);
```
solves the problem:
$$\text{fminbnd}(f,a,b) = \mathrm{argmin}_{x\in(a,b)} f$$
using the algorithm provided by Brent (1972).

### fminsearch
provided a continuous and finite function $f:\mathbb{R}\rightarrow\mathbb{R}$ which is not-necessarily continuous or finite at $\pm\infty$, we can attempt to find a local minimum of $f$ near $x_0$ ussually within a small number of iterations (and likely to converge quickly for strongly convex $f$). The method:
```cpp
double fminsearch(const function<double(double)>& f, double x0, double alpha=0);
```
solves the problem $\text{fminsearch}(f,x_0)=\mathrm{argmin}_{\text{near }x_0}f$ using the Nelder-Mead algorithm restricted to one dimension. The parameter `alpha` specifies an initial step size for the algorithm in the positive direction. If one is not provided (or a non-positive value is provided) then $\alpha=\max\{\epsilon,\epsilon\times\left|x_0\right|\}$ where $\epsilon\approx 10^{-8}$.

### Multivariate Minimization

The following optimizers inherit from the virtual class:
```cpp
class GradientOptimizer : public NonLinSolver {
    public:
    const vec& grad;
    explicit GradientOptimizer(double tol=1e-3, long maxiter=100, bool verbose=false);

    virtual void minimize(vec& x, const dFunc& f, const VecFunc& grad_f) = 0;
};
```
The function `minimize` solves the problem:
$$\min_x f(x)$$
And stores the solution in `x`. The input `grad_f`$=\nabla f$. The member `grad` is the gradient at the final point.

### Broyden–Fletcher–Goldfarb–Shanno algorithm
Uses the BFGS algorithm for minimization using the strong Wolfe conditions. This method uses symmetric rank 1 updates to the inverse of the hessian using the secant equation with the further constraint that the hessian remain symmetric positive definite.
```cpp
class BFGS : public GradientOptimizer {
    public:
    const arma::mat& inv_hessian;
    explicit BFGS(
        double tol=1e-3,
        long maxiter=100,
        bool verbose=false,
        double wolfe1=1e-4,
        double wolfe2=0.9
    );

    void use_cgd();
    void use_chol();

    void enable_finite_differences();
    void disable_finite_differences();

    void minimize(vec& x, const dFunc& f, const VecFunc& grad_f) override;
    void minimize(vec& x, const dFunc& f, const VecFunc& grad_f, const MatFunc& hessian);
};
```
The functions `enable/disable_finite_differences` allows the user to specify whether to approximate the initial hessian by finite diferences, which is disabled by default, so instead the initial hessian is set to the identity matrix.

The parameters `wolfe_c1`, `wolfe_c2` are the traditional paramters of the strong wolfe conditions:
$$f(x_k + \alpha_k p_k) \leq f(x_k) + c_1 \alpha_k p_k^T\nabla f(x_k)$$
and 
$$-p_k^T\nabla f(x_k + \alpha_k p_k) \leq -c_2 p_k^T\nabla f(x_k).$$
The default values are `wolfe_c1 = 1e-4` and `wolfe_c2 = 0.9`.

Similarly to Newton and Quasi-Newton methods the Hessian can be inverted iteratively or directly, this time using the Cholesky decomposition when the Hessian is positive definite. The method rarely requires matrix inversion, and ideally never inverts the Hessian at all.

**note:** like Broyden's method, `BFGS` stores the inverse hessian in memory, this may become inneficient in space and time when the problem is sufficiently large.

### Limited Memory BFGS
Uses the limited memory BFGS algorithm, which differs from BFGS by storing a limited number of previous values of `x` and `grad_f(x)` rather than a full matrix. The number of steps stored can be specified by `steps`.

```cpp
class LBFGS : public GradientOptimizer {
    public:
    explicit LBFGS(
        long steps=5,
        double tol=1e-3,
        long max_iter=100,
        bool verbose=false,
        double wolfe1=1e-4,
        double wolfe2=0.9
    );
    
    void minimize(vec& x, const dFunc& f, const VecFunc& grad_f) override;
};
```

### Momentum Gradient Descent
This class implements gradient descent in three ways.
* GD with line search for step size
* GD with constant step size
* GD with constant step size and Nestrov momentum.

```cpp
class MomentumGD :public GradientOptimizer {
    public:
    const double& damping_parameter;
    const double& step_size;

    explicit MomentumGD(double tol=1e-3, long maxiter=1000, bool verbose=false);

    void set_step_size(double alpha, double damping_p=0.90);

    void minimize(vec& x, const dFunc& f, const VecFunc& grad_f) override;
};
```
The parameter `damping_param` has a good explaination found in this [article](https://distill.pub/2017/momentum/) where the author refers to it as $\beta$. Setting this value to 0 is equivalent to traditional gradient descent.

The step size can be specified by `alpha` this can improve performance over the adaptive line minimization when the gradient is easy to evaluate but the may require more iterations until convergence.

Calling `set_step_size` and setting `alpha` and `damping_param` disables line minimization which is otherwise used by default.

### Nelder-Mead Gradient Free Minimization
The [Nelder-Mead method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method) is a derivative free method that constructs a simplex in n dimensions and iteratively updates its vertices in the direction where the function decreases in value. This method is ideal for low dimensional problems (e.g. 2,3, 10 dimensions, maybe not 100, though). The simplex is initialized using a guess $x_0$ of the argmin of the objective function, the vertices are initialized according to:
$$v_i = x_0 + \alpha u_i$$ 
where $u_i$ are columns of a random orthogonal matrix generated using:
```cpp
int n = x.n_elem;
mat U = orth(randn(n,n));
```
This is done to guarantee the simplex is spread out, moreover the direction is random (as oposed to, e.g., the coordinate directions) to avoid pathological cases.

```cpp
class NelderMead : public NonLinSolver {
    public:
    explicit NelderMead(double tol=1e-3, long maxiter=1000, bool verbose=false);

    void set_step_size(double s);
    void set_expansion_param(double e);
    void set_contraction_param(double c);
    void set_shrinking_param(double s);
    void set_initial_simplex_size(double s);

    void minimize(vec& x, const dFunc& f);
};
```
The function `set_step_size` sets the scaling of the reflection step, the default value is 1. The function `set_expansion_param` sets the scaling of the expanding step, the default value is 2. The function `set_contraction_param` sets the scaling of the contraction step, the default value is 0.5. The function `set_shrinking_param` sets the scaling of the shrinking step, the default value is 0.5. All of these parameters are explained [here](http://www.scholarpedia.org/article/Nelder-Mead_algorithm).

The parameter `initial_simplex_size` is the value $\alpha$ as desribed in the simplex initialization procedure discussed above.

### Genetic Maximization Algorithm
This method uses a genetic algorithm for _**maximization**_.
```cpp
class GeneticOptimizer : public NonLinSolver {
    public:
    explicit GeneticOptimizer(long seed=0, double tol=1e-1, long maxiter=100);

    void set_search_radius(double r);

    void set_population_size(long popsize);

    void set_diversity_parameters(
        double reproduction=0.5,
        double mutation=0.5,
        double diversity=0.2,
        long cutoff=30
    );

    void maximize(
        vec& x,
        const dFunc& f,
        const vec& lower_bound,
        const vec& upper_bound
    );
    void maximize(vec& x, const dFunc& f);
};
```
This method has a variety of parameters for updating the population of parameters to minimize with respect to:
* `population_size` : number of samples.
* `reproduction` : parameter for geometric probability distribution of "reproducing agents". i.e. if `reproduction` is close to 1, then only the most fit will reproduce and the algorithm will converge more quickly at the cost of possibly not optimal results (such a getting stuck at local maxima). If `reproduction` is close to 0, then most members will be able to participate at the cost of slower convergence. default value = 0.5.
* `mutation` : rate at which to introduce random perturbation to the population. Values close to 1 result in a population with higher variance resulting in slower convergence. Values close to 0 result in a population with higher variance resulting in faster convergence at the cost possible not optimal results (such as getting stuck at local optima).
* `diversity` : rate at which we encourage diversity. A higher value means agents further from optimality will have a higher change of reproduction which increases the chance of leaving local minima.
* `cutoff` : number of iterations after which we stop incentivising variance in the population. A lower value means quicker convergence at the cost possible not optimal results (such as getting stuck at local optima).

We can use this method for both box constrained and unconstrained maximization.

## Interpolation
### Polynomials
The following class implements real-valued algebraic polynomials which can be used for interpolation and approximation. The class implements basic vector space operations such as addition, negation, and multiplication; additionally, it implements derivatives and integrals.
```cpp
class Polynomial {
    public:
    const u_int& degree;
    const vec& coefficients;

    explicit Polynomial(double s=0);
    explicit Polynomial(const vec& p);
    explicit Polynomial(vec&& p);
    explicit Polynomial(const vec& x, const vec& y);
    explicit Polynomial(const vec& x, const vec& y, u_int deg);

    Polynomial(const Polynomial& P);
    void operator=(const Polynomial& P);

    double operator()(double x) const;
    vec operator()(const vec& x) const;

    Polynomial derivative(u_int k=1) const;
    Polynomial integral(double c=0) const;

    Polynomial operator+(const Polynomial& P) const;
    Polynomial operator+(double c) const;
    
    Polynomial operator-() const;
    Polynomial operator-(const Polynomial& P) const;
    Polynomial operator-(double c) const;

    Polynomial operator*(const Polynomial& P) const;
    Polynomial operator*(double c) const;

    Polynomial& operator+=(const Polynomial& P);
    Polynomial& operator+=(double c);
    Polynomial& operator-=(const Polynomial& P);
    Polynomial& operator-=(double c);
    Polynomial& operator*=(const Polynomial& P);
    Polynomial& operator*=(double c);
};
```
The polynomial can be constructed by setting it to a scalar, or by providing coefficients that are ordered:

$$p[0] x^{N-1} + p[1] x^{N-2} + \cdots + p[N-2] x + p[N-1].$$

i.e. the same ordering as in Armadillo's `polyfit` and `polyval` functions. Finally, the class can also be constructed by fitting a polynomial to a set of data (specifying `deg` sets the degree of the polynomial; if `deg` is not specified the it will be set to `x.n_elem-1` i.e. the interpolating polynomial).

This class is the base for the [`PieceWisePoly`](#Piecewise-Polynomials) class and is also an output of [`spectral_derivative`](#spectral-derivatives) and the solution form of [`BVPCheb`](#chebyshev-spectral-method).

If there is only a need to fit and interpolate a data set once, we may find it more efficient ($\mathcal O(n^3)\rightarrow\mathcal O(n^2)$) to interpolate using Lagrange interpolation:
```cpp
mat lagrange_interp(const vec& x, const mat& Y, const vec& xgrid);
```
where `xgrid` is the set of values to interpolate over. Note that this function is also vectorized to except a matrix `Y` and the interpolation is performed column-wise, i.e. `Y` must have the same number of rows as elements in `x`.

### Piecewise Polynomials
This class implements a data-structure used for storing and evaluating piece-wise polynomials defined on subinterval of a single connected interval. It does not implement its own constructor beyond a specifying an extrapolation method. 
```cpp
class PieceWisePoly {
    public:
    PieceWisePoly(const std::string& extrapolation="const", double val=0);

    double operator()(double t) const;
    vec operator()(const vec& t) const;

    PieceWisePoly derivative(int k=1) const;
    PieceWisePoly integral(double c=0) const;
};
```
The class implements an evaluation operation, and derivative and integral methods. The evaluation is well defined everywhere via the extrapolation method which is one of:
* `"const"` : constant value set by `val` on construction.
* `"boundary"` : when evaluated at a point less than the lower boundary, it evaluates to the same value as exactly on the boundary. Same for upper boundary.
* `"linear"` : linear exrapolation which assures continuity in the function and its first derivative.
* `"polynomial"` : uses the polynomial pieces at the boundaries for exrapolation.
* `"periodic"` : uses periodic extension.

The `derivative` function computes derivative to each piece. The `integral` function computes the integral of each piece but guarantees continuity between the pieces. If a `PieceWisePoly` object `ppoly` is defined on the interval $[a,b]$, then
$$\texttt{ppoly.integral(c)} = c + \int_a^x \texttt{ppoly}(\xi)d\xi$$

There are two functions currently implemented that actually construct `PieceWisePoly` objects. They are
```cpp
PieceWisePoly natural_cubic_spline(
    const vec& x, 
    const vec& y, 
    const string& extrapolation="boundary", 
    double val=0
);

PieceWisePoly hermite_cubic_spline(
    const vec& x,
    const vec& y,
    const string& extrapolation="linear",
    double val=0
);

PieceWisePoly hermite_cubic_spline(
    const vec& x,
    const vec& y,
    const vec& yp,
    const string& extrapolation="linear",
    double val=0
);
```
The function `natural_cubic_spline` implements a $C^2$ piecewise cubic interpolant of the data `(x,y)` with "natural boundary conditions", i.e. the second derivative is zero at both end-points of the interpolation range.

The function `hermite_cubic_spline` implements a $C^1$ piecewise cubic interpolant by specifying the desired derivative for every point as well, so it interpolates `(x,y,y')`. When `yp` is not specified, the derivative is inferred using first order finite differences (which is a classic Hermite spline).

### Sinc interpolation
Given sorted _**uniformly spaced**_ points on an interval, we can interpolate the data using a linear combination of sinc functions $\text{sinc}(x) = \frac{\sin(\pi x)}{\pi x}$:
```cpp
mat sinc_interp(const vec& x, const mat& Y, const vec& xgrid);
```
Note that this function is vectorized to except a matrix `Y` and the interpolation is performed column-wise, i.e. `Y` must have the same number of rows as elements in `x`.

## Namespace `neighbors`
This namespace holds utility classes for the more notable class `KDTree` which implements a batch initialized K-dimensional tree structure for $\mathcal{O}(\log n)$ nearest neighbors query.

```cpp
class neighbors::KDTree {
    public:
    const mat& data; // read only view of the data
    const u_long& p_norm;
    const u_long& leaf_size;

    explicit KDTree(int pnorm=2, int leafsize=30);
    explicit KDTree(const string& pnorm, int leafsize=30);
    KDTree(const KDTree& T);

    void fit(const mat& x);
    void fit(mat&& x);
    
    double min(u_long dim) const;
    double max(u_long dim) const;

    void query(mat& distances, umat& neighbors, const mat& pts, u_long k) const;
    void query(umat& neighbors, const mat& pts, u_long k) const;
};
```
The data structure is initialized by specifying a p-norm to induce a distance function between points, this parameter is any integer >= 1 or `"inf"` (i.e. a valid `p` value for Armadillo's `norm()`). Additionally, the leaf-size of the tree can be specified. It is often more efficient to perform nearest neighbor queries using the brute force method over a full tree traversal for small data sets so the tree structure terminates recursion once fewer than or equal to `leafsize` many points are under a node and stores those points in that node.

This class is the underlying data structure for [`KNeighborsEstimator`](#).

## Data Science & Machine Learning
Many of the models in this section inherit from the following classes:
```cpp
template<typename eT> class AutoEstimator {
    public:
    virtual void fit(const mat& x) = 0;
    virtual Col<eT> fit_predict(const mat& x) = 0;
    virtual Col<eT> predict(const mat& x) const = 0;
};

template<typename eT> class Estimator {
    public:
    virtual void fit(const mat& x, const Col<eT>& y) = 0;
    virtual Col<eT> predict(const mat& x) const = 0;
    virtual double score(const mat& x, const Col<eT>& y) const = 0;
};

typedef Estimator<double> Regressor;
typedef Estimator<arma::uword> Classifier;
```
Classes inheriting from `AutoEstimator` operate on a single variable to extract useful information from it. Classes inhering from `Estimator` operate on two variables to predict one from the other.

The score function returns a quality of estimate metric for data pairs `(x,y)` relative to the learned model.

### Feature Maps
#### Cubic Kernel
The following function implements the cubic kernel
$$k(x,\tilde x) = \|x - \tilde x\|_2^3$$
For use in the kernel trick.
```cpp
mat cubic_kernel(const mat& x);
mat cubic_kernel(const mat& x1, const mat& x2);
```
The first instance with only a single input computes the Gram matrix of the kernel for given data set which is symmetric and can therefore be computed more efficiently.

The second version computes a matrix which has `x1.n_rows` rows and `x2.n_rows` columns.

These two functions are used in the implementation of the [`Splines`](#splines) class.

#### Polynomial Features
The following class implements multi-dimensional polynomial terms for featurization of data. Due to the combinatory nature of multi-dimensional polynomials, the computational cost is exponential in the `degree` or highest order terms desired.
```cpp
class PolyFeatures {
    public:
    explicit PolyFeatures(int degree, bool include_intercept=false);

    void fit(const mat& x);
    mat fit_predict(const mat& x);

    mat predict(const mat& x) const;
};
```

### K-Fold Train Test Split
When performing cross validation we may want to split the data into training and testing subsets. The `KFolds1Arr` and `KFolds2Arr` procedure splits the data into `k` equal sized subsets of the data (randomly selected) for repeated training and testing.
```cpp
template<typename A> class KFolds1Arr {
    public:
    const Mat<A>& X;

    explicit KFolds1Arr(int n_folds=2);

    void fit(const Mat<A>& xx);
    viod fit(Mat<A>&& xx);

    Mat<A> train(u_long i);
    Mat<A> test(u_long i);
};

template<typename A, typename B> class KFolds2Arr {
    public:
    const Mat<A>& X;
    const Mat<B>& y;

    explicit KFolds2Arr(int n_folds=2);
    
    void fit(const aMat<A>& xx, const Mat<B>& yy);
    void fit(Mat<A>&& xx, Mat<B>&& yy);
    
    Mat<A> trainX(u_long i) const;
    Mat<B> trainY(u_long i) const;
    Mat<A> testX(u_long i) const;
    Mat<B> testY(u_long i) const;
};

typedef KFolds2Arr<double,double> KFolds;
```
The classes are initialized by specifying the number of splits. Calling `fit` copies or moves the arrays into the data-structure. Calling `train(i)`, `trainX(i)`, or `trainY(i)` returns the i-th training set of the respective arrays. Calling `test(i)`, `testX(i)`, or `testY(i)` returns the testing set of the i-th arrays. The const references `X` and `y` point to the fitted arrays in memory for read only use.

#### Example
```cpp
mat X = randu(100,1);
mat Y = 2*x;

KFolds train_test(X,Y,3);

int j = 0;
mat train_X =  train_test.trainX(j);
mat test_X = train_test.testX(j);

mat train_Y = train_test.trainY(j);
mat test_Y = train_test.testY(j);
```

### Label Encoding
Most classification models require labeled data to take the form of either integer labels or a one-hot encoding. The following classes transform data into these forms:
```cpp
template<class T> class LabelEncoder {
    public:
    LabelEncoder();

    void fit(const vector<T>& y);
    
    uvec encode(const vector<T>& y) const;
    vector<T> decode(const uvec& y) const;
};

class OneHotEncoder {
    public:
    explicit OneHotEncoder(bool plus_minus=false);

    void fit(const uvec& x);

    mat encode(const uvec& x) const;
    uvec decode(const mat& x) const;
};
```
The class `LabelEncoder` can be used for converting standard library vectors contain any object to integer labels. The class `OneHotEncoder` can be used for encoding integer labels into one-hot labels. The one-hot labels always encodes positive labels as `1`; by default negative labels are encoded as `0`, but can be set to `-1` if `plus_minus` is set to `true`. The first strategy is most often used for logistic regression and other cross entropy classifiers, while the latter is used for support vector machines.

### Data Binning
Whenever data is sampled from continuous features, it is sometimes useful (and easy) to bin the data into uniformly spaced bins. A primary benefit is reducing the complexity to $O(1)$ for any computation (number of bins is fixed by the user). In the univariate case, 500 bins is more than sufficient for large data.

The following class produces bins for univariate data.

```cpp
class BinData {
    public:
    const u_long& n_bins;
    const double& bin_width;
    const vec& bins;
    const vec& counts;

    explicit BinData(long bins=0);
    
    void fit(const mat& x);
    void fit(const mat& x, const vec& y);
};
```
We set the number of bins on initialization, but when `bins` is not specified, it is selected in `fit`. It will be set to 500 if the data size  `n>1000`, or `n/10` if `n/10 > 30`, and `n/5` in all other cases.

The member variable `bins` stores the bins corresponding to the `x` variable, and `counts` corresponds to the number of elements belonging to each bin using a linear weighting strategy. If fitted with only `x`, then
```cpp
sum(counts) == x.n_rows
```
And the result corresponds to a histogram. If fitted with both `x` and `y`, then
```cpp
sum(counts) == sum(y)
```
Where linear weights are used to average the values of `y` corresponding to each discrete value in `bins`. 

#### Example
```cpp
vec x = randn(200);
vec y = exp(-x);

int n_bins = 20;
BinData distribution(n_bins);
distribution.fit(x);
// if we plot distribution.bins and distribution.counts we should expect a histogram approximating a normal distribution

BinData discretized(n_bins);
discretized.fit(x,y);
// if we plot discretized.bins and discretized.counts we should expect a decaying exponentional
```

### K-Means Clustering
we can cluster unlabled data using Lloyd's algorithm accelerated by both kmeans++ for initialization and the triangle inequality for accelerating updates between iterations. Further speed-up can be attained using stochastic gradient descent (i.e. mini-batch kmeans) for very large data.
```cpp
class KMeans : public AutoEstimator<arma::uword> {
    public:
    const mat& clusters;
    const mat& cluster_distances;
    const mat& points_nearest_centers;
    const uvec& index_nearest_centers;
    
    explicit KMeans(int k, const string& pnorm, double tol=1e-2, long max_iter=100);
    explicit KMeans(int k, int p_norm=2, double tol=1e-2, long max_iter=100);

    virtual void fit(const mat& data) override;
    uvec predict(const mat& data) const override;
    virtual uvec fit_predict(const mat& data) override;
};

class KMeansSGD : public KMeans {
    public:
    explicit KMeansSGD(int k, int p_norm=2, int batch_size=100, double tol=1e-2, long max_iter=100);
    explicit KMeansSGD(int k, const string& p_norm, int batch_size=100, double tol=1e-2, long max_iter=100);

    void fit(const mat& data) override;
    uvec fit_predict(const mat& data) override;
};
```
The classes are both constructed by specifying the number of clusters, the distance measure, the tolerance for the stopping criteria ($\|\mu_{k+1} - \mu_k\| < \texttt{tol}$) and the maximum number of iterations the algorithm is permitted to run. For `KMeansSGD` we can also specify the batch size used in each iteration. The distance measure can be an integer >= 1 or `"inf"`.

The function `fit` performs the clustering on the specified data, and `predict` returns the row index in `clusters` corresponding to the nearest cluster center of each query point, i.e.
```cpp
mat nearest_center = km.clusters.rows(km.predict(x));
// nearest_center.row(i) is the nearest cluster to x.row(i)
```
Note that calling `fit_predict` is more efficient in this instance than calling `fit` then `predict`.

The member variables listed below are cached results from `fit`:
* `clusters` : `k` (number of clusters) by dimension. Each row is a cluster center.
* `cluster_distances` : `k` by `k`. Distances between the cluster centers.
* `points_nearest_centers` : `k` by dimension. The points in the data which are nearest to each cluster center, such as in a k-medioids problem.
* `index_nearest_centers` : `k` by 1. The locations in `data` corresponding to `points_nearest_centers`.

### Linear Models
Numerics implements a variety of linear models which inherit from the following class
```cpp
class LinearModel : public Regressor {
    public:
    const vec& linear_coefs;
    const double& intercept;
    const double& lambda;
    const double& eff_df;

    explicit LinearModel(bool intercept);
    void fit(const mat& X, const vec& y) = 0;
    vec predict(const mat& X) const = 0;
    double score(const mat& X, const vec& y) const = 0;
};
```
These models implement coefficient regularization with built-in cross-validation due to the efficiency of computation for these types of models. The member variable `linear_coefs` are the learned model coefficients, the variable `intercept` is the corresponding intercept term which is computed if `intercept=true` is set on construction. The variable `lambda` corresponds to the regularization term which typically appears as:
$$\texttt{loss}(\mathbf{w}) := \texttt{error}(y-X\mathbf{w}-b) + \lambda \cdot r(\mathbf{w}).$$
Where $r(\mathbf{w})$ is either the 1-norm or the 2-norm. The variable `eff_df` is the effective degrees of freedom in the system which is always less than or equal to `X.n_cols - 1`, this value is induced from a choice of `lambda`.

The `score` method of all linear models returns the coefficient of determination($R^2$) between `predict(X)` and `y` that are input into the function.

### LASSO Regression
For data `(x,y)`, the LASSO regression problem is as follows:
$$\min_{\mathbf{w},b} \|X\mathbf{w} + b - y\|_2^2 + \lambda\|\mathbf{w}\|_1$$
The parameter $\lambda$, in this implementation, is determined by K-folds cross validation, and is found efficiently using optimization techniques and a "warm start" between cross validation steps.

This implementation uses coordinate descent, and accepts a `tol` parameter and `max_iter` as stopping criteria for the optimization task.
```cpp
class LassoCV : public LinearModel {
    public:
    explicit LassoCV(bool intercept=true, double tol=1e-5, long max_iter=1000);

    void fit(const mat& X, const vec& y) override;
    vec predict(const mat& X) const override;

    double score(const mat& X, const vec& y) const override; // mse
};
```
For this class, `eff_df` is the number of non-zero coefficients and therefore always an integer value.

There is also a stand-alone function without cross validation:
```cpp
int coordinate_lasso(
    mat& w,
    const mat& X,
    const mat& y,
    double lambda,
    double tol=1e-4,
    u_long max_iter=1000,
    bool verbose=false
);

int coordinate_lasso(
    rowvec& b,
    mat& w,
    const mat& X,
    const mat& y,
    double lambda,
    double tol=1e-4, 
    long max_iter=1000, 
    bool verbose=false
);
```
Which solves the more general matrix problem, in the first version:
$$\min_W \|X W - Y\|_F^2 + \lambda \|W\|_{1,1}$$
and in the second version,
$$\min_{W,b} \|X W + \mathbf{1}b - Y\|_F^2 + \lambda \|W\|_{1,1}$$
This solver allows explicit selection of `lambda`, and can communicate progress by setting `verbose=true` which will display a progress bar and an exit message.

### Ridge Regression
For data `(X,y)`, the Ridge regression problem is
$$\min_{\mathbf{w},b} \|X \mathbf{w} + b - y\|_2^2 + \lambda \|\mathbf{w}\|_2^2$$
The following class finds an optimal solution to this problem by cross-validation:
```cpp
class RidgeCV : public LinearModel {
    public:
    const mat& cov_eigvecs;
    const vec& cov_eigvals;

    explicit RidgeCV(bool intercept=true);
    
    void fit(const mat& x, const vec& y) override;
    vec predict(const mat& x) const override;
    double score(const mat& x, const vec& y) const override;
};
```
The Ridge regression problem has an explicit solution
$$\tilde X = \begin{bmatrix} X & \mathbb{1} \end{bmatrix}.$$
$$\begin{bmatrix}\mathbf{w} \\ b \end{bmatrix}= (\tilde{X}^T \tilde{X} + \lambda I)^{-1} \tilde{X}^T y$$
`RidgeCV` solves the problem using the eigen-value decomposition of the matrix:
$$\tilde{X}^T \tilde{X} = V \Gamma V^T$$
which results in the solution:
$$\begin{bmatrix}\mathbf{w} \\ b \end{bmatrix}= V(\Gamma + \lambda I)^{-1}V^T\tilde{X}^T y.$$
Which is efficient to compute for many values of $\lambda$ as is neccessary in the cross-validation step where `RidgeCV` determines an optimal $\lambda$ using the generalized cross validation score (GCV):
$$\text{GCV}(\lambda) = \frac{n}{(n-\text{df})^2} \|X \mathbf{w}_\lambda + b_\lambda - y\|_2^2$$
Which approximates the leave-one-out CV scheme without recomputing $w$ and $b$ for each train-test set. The parameter $\text{df}$ are the degrees of freedom in the model which is computed by:
$$\text{df} = \sum_{i=1}^k \frac{\gamma_i}{\gamma_i + \lambda}$$
where $\gamma_i$ are the eigenvalues from the matrix $\Gamma$, and $k$ is the dimension of $w$ plus 1. For this class, `eff_df` is not necessarily an integer.

### Splines
Fit a nonlinear model to data `(X,y)` using the kernel trick, specifically, the [cubic kernel](#cubic-kernel). The generated model solves the problem:
$$\min_{\mathbf{w,c},b} \|X\mathbf{w} + k(X,X)\mathbf{c} + b - y\|_2^2 + \lambda\|\mathbf{c}\|_2^2$$
Where $k$ is the cubic kernel and $k(X,X)$ is the Gram matrix of the kernel. As in [`RidgeCV`](#ridge-regression) $\lambda$ is determined by GCV and made more efficient with the eigenvalue decomposition of the Gram matrix.
```cpp
class Splines : public LinearModel {
    public:
    const vec& kernel_coefs;
    const mat& X;
    const mat& kernel_eigvecs;
    const vec& kernel_eigvals;

    explicit Splines();

    void set_lambda(double l);
    void set_df(double df);

    void fit(const mat& x, const vec& y) override;
    arma::vec predict(const mat& x) const override;
    double score(const mat& x, const vec& y) const override;    
};
```
Since the number of features scales with the number of data points, it may be expensive to search for an optimal $\lambda$ so the functions `set_lambda` and `set_df` enable the user to specify the regularization term which may be informed from either $\lambda$ or the desired degrees of freedom of the model.

### Logistic Regression
Fit a linear logistic regression model for data `(X,y)`. By default, the regression is done via L2-regularization and the regularization parameter is determined by cross validation but can be set with `set_lambda`. The model minimizes a regularized log-likelihood:
$$\min_{W,b} \lambda\|W\|_F^2 -\sum_{i=1}^n y^T_i\log\sigma(x_i W + \mathbf{1}b).$$
Where $\sigma$ is the softmax, and $y_i$ is the onehot encoding of classes. The logarithm is interpreted element-wise. The resulting model is a one vs. all probabilistic model, which predicts the class label by the maximum likelihood. When fitting the model, an [`LBFGS` solver](#limited-memory-bfgs) solver is used to solve the minimization problem.
```cpp
class LogisticRegression : public Classifier {
    public:
    const double& lambda;
    const mat& linear_coefs;
    const rowvec& intercepts;
    const OneHotEncoder& encoder;

    explicit LogisticRegression();

    void set_lambda(double l);

    void fit(const mat& x, const uvec& y) override;
    uvec predict(const mat& x) const override;
    mat predict_proba(const mat& x) const;
    double score(const mat& x, const uvec& y) const override;
};
```
The model requires integer labels for fitting, and stores an instance of a `OneHotEncoder` in memory which can be refereced via `encoder`. The `predict` method returns predicted integer labels, and `predict_proba` returns the softmax output which can be interpreted as the likelihood of each class given `x`. The `score` function evaluates the accuracy of `predict(x)` relative to `y`.

### Neural Networks and SGD Models
The following classes inherit from
```cpp
class ModelSGD {
    public:
    explicit ModelSGD(
        const string& loss,
        long max_iter,
        double tol,
        double l2,
        double l1,
        const string& optimizer,
        bool verbose
    );
};
```
These models implement a [neural-network](#) which can be interfaced with a `fit` method inherited from an `Estimator` type. These models all solve the following problem:
$$\min_{W} \texttt{loss}(f(X;W), y) + \texttt{l1}\|W\|_F^2 + \texttt{l2}\|W\|_{1,1}.$$
The model parameters are not necessarily a single matrix as perhaps suggested here, but the interpretation of the loss functions is the same. The parameters `l2` and `l1` should therefore be non-negative values. The parameters `loss` and `optimizer` have an explanation in the [neural-network](#) section. The parameter `max_iter` is the maximum iteration for the solver, and `verbose` enables or disables communication with the solver which displays a progress bar as well as an exit message.

The model solves the minimization problem using stochastic gradient descent (or one of its variants). As a result, the model training procedure is fast, but not-necessarily as optimal as a model trained with the full data at each step of the optimization procedure.

### Linear SGD Models
There are two linear SGD models, first linear regression:
```cpp
class LinearRegressorSGD : public Regressor, public ModelSGD {
    public:
    explicit LinearRegressorSGD(
        const string& loss="mse",
        long max_iter=200,
        double tol=1e-4,
        double l2=1e-4,
        double l1=0,
        const string& optimizer="adam",
        bool verbose=false
    );

    void fit(const mat& x, const vec& y) override;
    vec predict(const mat& x) const override;
    double score(const mat& x, const vec& y) const override;
};
```
This class implements a regularized linear model whose parameters are trained via stochastic gradient descent. The accepted losses are either `mse` (mean-squared-error) in which case the model minimizes:
$$\min_{\mathbf{w},b} \|X\mathbf{w}+b-y\|_2^2 + \texttt{l2}\|\mathbf{w}\|_2^2 + \texttt{l1}\|\mathbf{w}\|_1$$
or `mae` (mean-absolute-error) in which case the model minimizes:
$$\min_{\mathbf{w},b} \|X\mathbf{w}+b-y\|_1 + \texttt{l2}\|\mathbf{w}\|_2^2 + \texttt{l1}\|\mathbf{w}\|_1.$$
The model's `score` is the coefficient of determination ($R^2$).

The second linear model is a linear classifier:
```cpp
class LinearClassifierSGD : public Classifier, public ModelSGD {
    public:
    explicit LinearClassifierSGD(
        const string& loss="categorical_crossentropy",
        long max_iter=200,
        double tol=1e-4,
        double l2=1e-4,
        double l1=0,
        const string& optimizer="adam",
        bool verbose=false
    );

    void fit(const mat& x, const uvec& y) override;

    mat predict_proba(const mat& x) const;
    uvec predict(const mat& x) const override;
    double score(const mat& x, const uvec& y) const override;
};
```
This class implements a regularized linear classifier whose parameters are trained via stochastic gradient descent. Currently, the only loss function is `categorical_crossentropy` so the model minimizes:
$$\min_{W,b} \texttt{l2}\|W\|_F^2 + \texttt{l1}\|W\|_{1,1} -\sum_{i=1}^n y^T_i\log\sigma(x_i W + \mathbf{1}b).$$
Behaves just as [logistic regression](#logistic-regression) in terms of `predict`, `predict_proba`, and `score`.

### Neural Network Models
Another set of models trained using SGD are neural-network models. These models use a set of linear layers with non-linear activations functions to predict non-linear trends in data. Just as with the linear SGD models, there are also two neural network models. First the regressor:
```cpp
class NeuralNetRegressor : public Regressor, public ModelSGD {
    public:
    explicit NeuralNetRegressor(
        const vector<pair<int,string>>& layers={{100,"relu"}},
        const string& loss="mse",
        long max_iter=200,
        double tol=1e-4,
        double l2=1e-4,
        double l1=0,
        const string& optimizer="adam",
        bool verbose=false
    );

    void fit(const mat& x, const vec& y) override;
    vec predict(const mat& x) const override;
    double score(const mat& x, const vec& y) const override;
};
```
In addition to the inputs accepted by `ModelSGD`, the hidden layer design should also be specified to the model in construction. This is achieved by providing a standard `vector` of `pair`s where `first` is an `int` specifying the number of nodes in the layer and `second` is a `string` specifying the [activation function](#). Note that this is only the design of the hidden layers, by default, the model constructs the input and output layers based on `x` and `y` during fit. So by default the model one hidden layer with 100 nodes and the [ReLU](#) activation, i.e.:
$$\texttt{predict}(X) = \texttt{relu}(XW_1 + b_1)W_2 + b_2$$

The classifier:
```cpp
class NeuralNetClassifier : public Classifier, public ModelSGD {
    public:
    explicit NeuralNetClassifier(
        const vector<pair<int,string>>& layers={{100,"relu"}},
        const string& loss="categorical_crossentropy",
        long max_iter=200,
        double tol=1e-4,
        double l2=1e-4,
        double l1=0,
        const string& optimizer="adam",
        bool verbose=false
    );

    void fit(const mat& x, const uvec& y) override;
    mat predict_proba(const mat& x) const;
    uvec predict(const mat& x) const override;
    double score(const mat& x, const uvec& y) const override;
};
```
This model acts essentially like the regressor but predicts instead the one-hot encoding of `y` and maps the output to probabilities via the softmax function. Thus, the default model is:
$$\texttt{predict\_proba}(X) = \sigma\left(\texttt{relu}(XW_1 + b_1)W_2 + b_2\right).$$

### k Nearest Neighbors Estimators
Numerics implements a kNN regressor and classifier which inherit from the following virtual class:
```cpp
template<typename eT> class KNeighborsEstimator : public Estimator<eT> {
    public:
    const u_int& k;
    const arma::uvec& ks;
    const arma::vec& scores;
    const neighbors::KDTree& X;
    const arma::Col<eT>& y;

    explicit KNeighborsEstimator(
        int K,
        int p_norm,
        bool use_distance_weights,
        int leaf_size
    );

    explicit KNeighborsEstimator(
        int K,
        const string& p_norm,
        bool use_distance_weights,
        int leaf_size
    );

    explicit KNeighborsEstimator(
        uvec& Ks,
        int p_norm,
        bool use_distance_weights,
        int leaf_size
    );

    explicit KNeighborsEstimator(
        uvec& Ks,
        const string& p_norm,
        bool use_distance_weights,
        int leaf_size
    );

    void fit(const arma::mat& xx, const arma::Col<eT>& yy) override;
    void fit(arma::mat&& xx, arma::Col<eT>&& yy);
    arma::Col<eT> predict(const arma::mat& xx) const;
    virtual double score(const arma::mat& xx, const arma::Col<eT>& yy) const = 0;
};
```
This class implements inference from nearest neighbors using the [`KDTree` class](#) for fast queries. When predicting there are two options (1) average value of the k nearest neighbors or (2) an inverse distance weighing of the nearest neighbors (set `use_distance_weights=true`). The number of nearest neighbors is specified by `K` or `Ks`. If more than one value is provided, then one will be selected by cross-validation. Additionally, the `p_norm` for computing distances can be provided as well as the `leaf_size` of the `KDTree`. Since the data must be referenced for queries, the solver can be fit by copying the data or moving it.

### k Nearest Neighbors Classifier
The classifier is defined:
```cpp
class KNeighborsClassifier : public KNeighborsEstimator<uword> {
    public:
    explicit KNeighborsClassifier(
        int K,
        int p_norm=2,
        bool use_distance_weights=false,
        int leaf_size=30
    );
    
    explicit KNeighborsClassifier(
        int K,
        const string& p_norm,
        bool use_distance_weights=false, 
        int leaf_size=30
    );

    explicit KNeighborsClassifier(
        uvec& Ks,
        int p_norm=2,
        bool use_distance_weights=false,
        int leaf_size=30
    );
    
    explicit KNeighborsClassifier(
        uvec& Ks,
        const string& p_norm,
        bool use_distance_weights=false,
        int leaf_size=30
    );
    
    double score(const mat& xx, const uvec& yy) const override;
    mat predict_proba(const mat& xx) const;
};
```
It differs from its parent only in its implementation `score` (which is the accuracy), `predict` (though via a private member rather than by override), and `predict_proba`.

### k Nearest Neighbors Regressor
The regressor is defined:
```cpp
class KNeighborsRegressor : public KNeighborsEstimator<double> {
    public:
    explicit KNeighborsRegressor(
        int K,
        int p_norm=2,
        bool use_distance_weights=false,
        int leaf_size=30
    );

    explicit KNeighborsRegressor(
        int K,
        const string& p_norm,
        bool use_distance_weights=false,
        int leaf_size=30
    );

    explicit KNeighborsRegressor(
        const uvec& Ks,
        int p_norm=2,
        bool use_distance_weights=false,
        int leaf_size=30
    );
    
    explicit KNeighborsRegressor(
        const uvec& Ks,
        const string& p_norm,
        bool use_distance_weights=false,
        int leaf_size=30
    );
    
    double score(const arma::mat& xx, const arma::vec& yy) const override;
};
```
It differs from its parent onl in its implementation of `score` (which is R2), and `predict` (though via a private member rather than by override).

### Kernel Estimators
The following classes inherit from the class
```cpp
class KernelEstimator {
    public:
    const mat& X;
    const BinData& bin_data;
    const double& bandwidth;

    explicit KernelEstimator(const string& kernel, bool binning, long n_bins);
    explicit KernelEstimator(double bdw, const string& kernel, bool binning, long n_bins);
};
```
These classes implement regression tasks by weighing neighbor information via a kernel. These kernels are radial basis functions $K(x,\tilde x) = k\left(\frac{\|x-\tilde x\|}{\beta}\right)$ for a parameter $\beta$ which we call the bandwidth of $k$. The following kernels are implemented:
* `"gaussian"`: $k(r) = \frac{1}{\sqrt{2\pi}}e^{-r^2/2}$
* `"square"`: $k(r) =0.5I_{r\leq 1}(r)$
* `"triangle"`: $k(r) = (1-r)I_{r\leq 1}(r)$
* `"parabolic"`: $k(r) = \frac{3}{4}(1-r^2)I_{r\leq 1}(r)$

The bandwidth parameter controls the radius of information, so a small value corresponds to a very wide radius whereas a large value corresponds to a very narrow radius. When not specified, the bandwidth is infered using a variety of strategies specific to each estimator.

Finally, the data may also be binned to a specified number of bins as specified by setting `binning=true` and setting `n_bins` to the desired quantity (the latter is ignored if `binning=false`). We can make kernel smoothing more efficient (effectively $O(1)$) for large data by first binning the data (linear binning). We compute the kernels with respect to the bins rather than the observations without reducing the quality of fit significantly. For large data sets Gramacki (2018) argues that 400-500 bins is almost always sufficient for univariate distributions.

There is also a small namespace `bw` which implements basic 1D kernel estimation methods:
```cpp
namespace bw {
    vec eval_kernel(const vec& r, const string& K="gaussian");
    double dpi(const vec& x, double s=0, const string& K="gaussian");
    double dpi_binned(const BinData& bins, double s=0, const string& K="gaussian");
    double rot1(int n, double s);
    double rot2(const vec& x, double s = 0);
    double grid_mse(
        const vec& x,
        const string& K,
        double s=0,
        int grid_size=20,
        bool binning=false
    );
    double grid_mse(
        const vec& x,
        const vec& y,
        const string& K,
        double s=0,
        int grid_size=20,
        bool binning=false
    );
}
```
* `eval_kernel` : evaluates any of the implemented kernels for values of `r`.
* `dpi(x, s=0, K="gaussian")` : compute the direct plug in (L=2) bandwidth estimate for kernel density estimation.
  * x : data to estimate bandwidth for.
  * s : precomputed standard deviation of x, if s <= 0, then s will be computed.
  * K : kernel.
* `dpi_binned(bins, s=0,  K=gaussian)` : compute the direct plug in (L=2) bandwidth estimate for kernel density estimation for pre-binned data.
  * bins : [BinData object](#data-binning) of prebinned data.
  * s : precomputed standard deviation of data, if s <= 0, then s will be computed.
  * K : kernel.
* `rot1(n, s)` : the original rule of thumb bdw = 1.06 * s * n^(-1/5). Originally proposed by Silverman (1986) is optimal whenever the true distribution is normal.
  * n : size of data.
  * s : standard deviation of data.
* `rot2(x, s=0)` : a more common rule of thumb bdw = 0.9 * min(IQR/1.34, s) * n^(-1/5). More robust at treating mixture models and skew distributions.
  * x : data to compute bandwidth estimate for.
  * s : precomputed standard deviation of data, if s <= 0, then s will be computed.
* `grid_mse(x, K, s=0, grid_size=20, binning=false)` : compute an optimal bandwidth for kernel density estimation by grid search cross-validation using an approximate RMSE, where the true density is estimated using a pilot density. The pilot density is computed using the entire data set and the bdw_rot2 estimate for the bandwidth. Then the MSE for each bandwidth is computed by predicting the density for a testing subset of the data computed using the rest of the data.
  * x : data to estimate bandwidth for.
  * K : kernel to use.
  * s : precomputed standard deviation, if s <= 0, then s will be computed.
  * grid_size : number of bandwidths to test, the range being [0.05*s, range(x)/4] using log spacing.
  * binning : whether to prebin the data. Typically the estimate is just as good, but the computational cost can significantly reduced.
* `grid_mse(x, y, K, s=0, grid_size=20, binning=false)` : compute an optimal bandwidth for kernel smoothing by grid search cross-validation using an approximate RMSE.
  * x : independent variable.
  * y : dependent variable.
  * K : smoothing kernel.
  * s : precomputed standard deviation of x, if s <= 0, then s will be computed.
  * binning : whether to bin the data or not.

### Kernel Density Estimation
The density function of random data can be estimated by a weighted sum of kernels centered on the data and evaluated at a query point. The class `KDE` implements density estimation for one-dimensional data.
```cpp
class KDE : public KernelEstimator, public AutoEstimator<double> {
    public:
    explicit KDE(
        const string& kernel="gaussian",
        const string& bandwidth_estimator="min_sd_iqr",
        bool binning=false,
        long n_bins=30
    );

    explicit KDE(
        double bdw,
        const string& kernel="gaussian",
        bool binning=false,
        long n_bins=30
    );

    void fit(const mat& x) override;
    vec fit_predict(const mat& x) override;
    vec predict(const mat& x) const override;
    vec sample(int n) const;
};
```
The model is initialized by specifying a kernel as described in [kernel estimators](#kernel-estimators). Additionally, the bandwidth can be specified via `bdw` or a `bandwidth_estimator` strategy can be specified which is one of:
* `rule_of_thumb` : corresponds to `bw::rot1`.
* `min_sd_iqr` : corresponds to `bw::rot2`.
* `plug_in` : corresponds to `bw::dpi`.
* `grid_cv` : corresponds to `bw::grid_mse`.

The `predict` method returns the probability density at query points (rows of `x`).

The `sample` method returns `n` random samples from the distribution. The sampling is done randomly sampling the data (with replacement) and subsampling from each of those points according to the CDF defined by the kernels centered at those points.

### Kernel Smoothing
Kernel estimation may be applied to the task of non-linear regression. The class `KernelSmooth` implements such an estimator for one-dimensional data.
```cpp
class KernelSmooth : public KernelEstimator, public Regressor {
    public:
    const mat& y;

    explicit KernelSmooth(
        const string& kernel="gaussian",
        bool binning=true,
        long n_bins=30
    );
    
    explicit KernelSmooth(
        double bdw,
        const string& kernel="gaussian",
        bool binning=true,
        long n_bins=30
    );
    
    void fit(const mat& X, const vec& y) override;
    vec predict(const mat& X) const override;
    double score(const mat& x, const vec& y) const override;
};
```
The model is initialized by specifying a kernel as described in [kernel estimators](#kernel-estimators). A bandwidth can be specified with `bdw` or one is selected with `bw::grid_mse`.

The `score` method evaluates the $R^2$ between `predict(x)` and `y`. 

# `numerics::ode` Documentation
## Table of Contents
* [`numerics.hpp` documentation](#`numerics.hpp`-documentation)
* [differential operators](#differential-operators)
* [Initial Value Problems](#Initial-Value-Problem-Solvers)
    * [Dormand-Prince 4/5](#dormand-prince-4/5)
    * [Runge-Kutta explicit $\mathcal O(4)$](#runge-kutta-fourth-order)
    * [Runge-Kutta adaptive implicit $\mathcal O(4)$](#Runge-Kutta-Implicit-Fourth-Order)
    * [Runge-Kutta implict $\mathcal O(5)$](#Runge-Kutta-Implicit-Fifth-Order)
    * [backwards Euler](#backwards-euler)
    * [Adams-Moulton second order](#adams-moulton-second-order)
    * [event control](#ivp-events)
* [Boundary Value Problems](#boundary-value-problems-solver)
    * [variable order method]()
    * [spectral method]()
    * [Lobatto IIIa]()
* [Poisson's Equation](#poisson-solver)

The following are all members of namespace `numerics::ode`.

## Differentiation Operators
Given an interval $\Omega=[L,R]$, if we sample $\Omega$ at points $x = \{x_1, \cdots, x_N\}$ we can approximate the continuous operator $\frac{d}{dx}$ at the sampled points with discrete operator $D$. This operator can be applied to any differentiable $f:\Omega\rightarrow\mathbb{R}$ given the function values at the sample points: $y = \{f(x_1),\cdots,f(x_N)\}$ according to: $f'(x) \approx D y$.
```cpp
void diffmat4(arma::mat& D,
              arma::vec& x,
              double L, double R,
              unsigned int sample_points);
void diffmat2(arma::mat& D,
              arma::vec& x,
              double L, double R,
              unsigned int sample_points);
void cheb(arma::mat& D,
          arma::vec& x,
          double L, double R,
          unsigned int sample_points);
void cheb(arma::mat& D, arma::vec& x, unsigned int sample_points);
```
In all of these functions the discrete operator is assigned to `D`, and the sample points are assigned to `x`. The parameters `L` and `R` define the end points of the interval $\Omega$. The parameter `sample_points` defines how many points to sample from the interval.

The function `diffmat4` samples the interval uniformly and provides a fourth order error term i.e. error is $\mathcal{O}(N^{-4})$. The resulting operator has a bandwidth of 4. It is also the case that the eigenvalues of $D$ are all of the form $\lambda_k=-b_ki$ where $b_k\geq 0$ and $i=\sqrt{-1}$.

The function `diffmat2` samples the interval uniformly and provies a second order error term i.e. error is $\mathcal{O}(N^{-2})$. The resulting operator has a bandwidth of 2. It is also the case that the eigenvalues of $D$ are all of the form $\lambda_k=-b_ki$ where $b_k\geq 0$ and $i=\sqrt{-1}$.

The function `cheb` samples the interval at Chebyshev nodes and converges spectrally. If no interval is provided, the interval is set to $[-1,1]$. The resulting operator is dense. Moreover $(D_{\text{cheb}})^k = \frac{d^k}{dx^k}$.

In all three cases the $n\times n$ operator has rank $n-1$ which follows from the intuition that the null space of the derivative is the set of all (piecewise-)constant functions.

A more generic differentiation matrix is offered by the following two functions:
```cpp
arma::rowvec diffvec(const arma::vec& x, double x0, unsigned int k=1);

diffmat(arma::mat& D, const arma::vec& x, unsigned int k=0, unsigned int bdw=2);

diffmat(arma::sp_mat& D, const arma::vec& x, unsigned int k=0, unsigned int bdw=2);
```
Where `diffvec` returns a rowvector $\vec d_k$ such that $\vec d_k\cdot f(\vec x) \approx f^{(k)}(x_0)$.

The function `diffmat` produces a differentiation matrix $D$ for any grid of points $x$ (does not need to be uniform or sorted) such that $D_k f(x) \approx f^{(k)}(x)$. Setting the `bdw` parameter will allow the user to select the number of nearby points to use in the approximation, for example if `bdw=2` then for $x_i$, the points $\{x_{i-1},x_i,x_{i+1}\}$ will be used in the approximation. Moreover, expect the error in the approximation to be $\mathcal O(h^{\text{bdw}})$ where $h$ is the maximum spacing in $x$. A special benefit of `diffmat` is that we can find any order derivative, moreover for any $n\times n$ $D_{k}$ we have $\text{rank}(D_k) = n-k$, and the eigenvalues are of the form $\lambda = i^kb$ where $b \geq 0$ and $i = \sqrt{-1}$. (so $D_2$ is positive semi-definite for example).

Given a linear ODE of the form: $y' + \alpha y = f(x)$ and the initial condition: $y(L) = \beta$, we can approximate the solution by solving the linear system: $(D+\alpha I)y = f(x) \land y(L) = \beta$. This can be solved by forward substituting $y(L) = \beta$ into the linear system and solving the rest of the system:
```cpp
vec f(vec& x) {
    // do something
}
mat D;
vec x, y;
double L, R, alpha, beta;
int N;
int method; // set to 1 or 2

diffmat2(D,x,L,R,N); // or diffmat4(), or cheb(), or diffmat()

mat A = D.rows(1,N-1).cols(1,N-1) + alpha*eye(N-1,N-1);
vec d0 = D.col(0);
vec F = f(x.rows(1,N-1)) - d0.rows(1,N-1)*beta;

vec y(N);
y.rows(1,N-1) = solve(A,F);
```
If we have a system of $m$ ODEs, we can solve both initial value problems and boundary value problems using a similar method where instead the operator is replaced with $(D \otimes I_{m,m})$ ($\otimes$ is the Kronecker product) and $f(x)$ is vectorized (if `F` is $n\times m$ then set `F = vectorise(F.t())`). Once a solution $y$ is found it is reshaped so that it is $n\times m$ (if `y` is $n m\times 1$, then set `y = reshape(y,m,n).t()`). For these higher order system both initial value problems and boundary value problems may be solved.

## Initial Value Problem Solvers
We define a system of initial value problem as having the form: $u' = f(t,u)$ with $u(0) = u_0$. Where $t$ is the independent variable and $u(t)$ is the dependent variable and is a row vector. All of the systems solvers are able to handle events. Some of the solvers have error control via adaptive step size selection. For the implicit solvers we can also provide a jacobian matrix $\frac{\partial f}{\partial u}$ to improve solver performance. All implicit solvers use Broyden's method or Newton's method to compute steps.

All solver inherit from the `ivp` class:
```cpp
class ivp {
    public:
    unsigned int max_nonlin_iter;
    double max_nonlin_err;
    unsigned int stopping_event;
    
    void add_stopping_event(const std::function<double(double,const arma::rowvec&)>& event,
                            event_direction dir = ALL);
};
```
Events are defined in a [later section](#ivp-events).

All IVP solvers have a function `ode_solve` which, at a higher level, describes the problem to solve and the solution (recall we are solving $u'=f(t,u), u(t_0)=u_0$):
```cpp
void ode_solve(f [,J], t, U);
```

The parameter `f` is $f(t,u)$.

If `J` is provided, it is the jacobian matrix: $J(t,u) = \frac{\partial f}{\partial u}$. The jacobian may be provided to the implicit solvers, though they perform comparably without.

The parameter `t` must be initialized to `{t_initial, t_final}`, this parameter will be overwritten with the grid points selected by the solver on the interval.

The parameter `U` should be initialized to $u_0$ as a single row vector. This parameter will be overwritten by the solver so that the pair {`t(i), U.row(i)`} is {$t_i$,$u(t_i)$}.

### Dormand-Prince 4/5
Fourth order explicit Runge-Kutta solver with adaptive step size for error control.
```cpp
class rk45 : public ivp {
    public:
    double adaptive_step_min; // 0.05
    double adaptive_step_max; // 0.5
    double adaptive_max_err; // tol
    rk45(double tol = 1e-3);
    void ode_solve(
        const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
        arma::vec& t,
        arma::mat& U);
};
```

### Runge-Kutta Fourth Order
classical Fourth order explicit Runge-Kutta solver with constant step size.
```cpp
class rk4 : public ivp {
    public:
    double step; // step_size
    rk4(double step_size = 0.1);
    void ode_solve(
        const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
        arma::vec& t,
        arma::mat& U);
};
```
### Runge-Kutta Implicit Fourth Order
Fourth order diagonally implicit Runge-Kutta solver with adaptive step size controlled via a third order approximation. Method is A-stable and L-stable.
```cpp
class rk45i : public ivp {
    public:
    double adaptive_step_min;
    double adaptive_step_max;
    double adaptive_max_err; // tol
    rk45i(double tol = 1e-3);
    void ode_solve(
        const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
        arma::vec& t, 
        arma::mat& U);
    void ode_solve(
        const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
        const std::function<arma::mat(double,const arma::rowvec&)>& jacobian,
        arma::vec& t,
        arma::mat& U);
};
```
This object accepts a jacobian matrix.

### Runge-Kutta Implicit Fifth Order
Fifth order semi-implicit Runge-Kutta solver with constant step size. Method is A-stable and L-stable.
```cpp
class rk5i : public ivp {
    public:
    double step;
    rk5i(double step_size = 0.1);
    void ode_solve(
        const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
        arma::vec& t,
        arma::mat& U);
    void ode_solve(
        const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
        const std::function<arma::mat(double,const arma::rowvec&)>& jacobian,
        arma::vec& t,
        arma::mat& U);
};
```
This object accepts a jacobian matrix.

### Backwards Euler
First order implicit Euler's method with constant step size. (Euler's method is not accurate but very stable):
```cpp
class am1 : public ivp {
    public:
    double step; // step_size
    am1(double step_size = 0.1);
    void ode_solve(
        const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
        arma::vec& t,
        arma::mat& U);
    void ode_solve(
        const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
        const std::function<arma::mat(double, const arma::vec&)>& jacobian,
        arma::vec& t,
        arma::mat& U);
};
```
This object accepts a jacobian matrix.

### Adams-Moulton Second Order
Second order implicit linear multistep method with constant step size. (Will likely be replaced by a B-stable alternative)
```cpp
class am2 : public ivp {
    public:
    double step; // step_size
    am2(double step_size = 0.1);
    void ode_solve(
        const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
        arma::vec& t,
        arma::mat& U);
    void ode_solve(
        const std::function<arma::rowvec(double,const arma::rowvec&)>& f,
        const std::function<arma::mat(double, const arma::rowvec&)>& jacobian,
        arma::vec& t,
        arma::mat& U);
};
```
This object accepts a jacobian matrix.

### IVP Events
The function:
```cpp
enum class event_direction {
    NEGATIVE = -1,
    ALL = 0,
    POSITIVE = 1
};
void add_stopping_event(const std::function<double(double,const arma::rowvec&)>& event,
                        event_direction dir);
```
Allows the the user to add an event function which acts as a secondary stopping criterion for the solver. An event function specifies to the solver that whenever $\text{event}(t_k,u_k) = 0$ the solver should stop. We can further constrain the stopping event by controlling the sign of $\text{event}(t_{k-1},u_{k-1})$. e.g. if `dir = NEGATIVE`, the solver will stop iff: $\text{event}(t_k,u_k) = 0$ __*and*__ $\text{event}(t_{k-1},u_{k-1}) < 0$.

The `ivp` member `stopping_event` will be set to the event index (of the events added) that stopped it. e.g. if the third event function added stops the solver, then `stopping_event = 2`.

## Boundary Value Problems Solver
We can solve boundary value problems using finite difference methods. A procedure for simple linear problems was described in the [operators section](#differentiation-operators), but the following methods are far more generalized. Our problem is defined as follows:

Given interval domain $\Omega = [L,R]$, and _**system**_ of ODEs $u' = f(x,u)$ with boundary conditions $g(u)=0$ on $\partial\Omega$ which is equivalently defined: $g(u(L),u(R)) = 0$. This general problem is solved, if possible, using Newton's method which requires an initial guess of the solution. One method for providing an initial guess is by solving the linearized problem $u' = \big(\frac{\partial f}{\partial u}\big|_{u=u_0}\big)\cdot u$ where $u_0$ should be either $u(L)$ or $u(R)$. Moreover, it is ideal if the initial function satisfies the boundary conditions.

For example, given interval $\Omega=[0,1]$. if one of the equations is $u'' = sin(u)$ with boundary condition $u(0)=1$ and $u(1)=0$. Set up the problem as a system of first order ODEs: $u' = v$ and $v' = sin(u)$, with the same boundary conditions. Then, solve instead $u' = v \land v'=(\frac{d}{du}sin(u)\big|_{u=1})u = cos(1)u$. The linearized solution is then $u(x) = b(e^{-a(x-2)} - e^{ax}) \land v(x) = -ba(e^{a(x-2)}+e^{ax})$ where $a=\sqrt{\cos 1}, b = e^{2\sqrt a} - 1$.

The `bvp` class:
```cpp
class bvp {
    public:
    unsigned int max_iterations; // maximum iterations for Newton's method
    double tol; // stopping criteria
    int num_iterations(); // returns the number of iterations needed by the solver
};
```

### Variable Order Method
We first introduce a method finite differencing where the order of the polynomial interpolation from the difference formula is derived may be selected by the user. For uniformly spaced data, the method should be $\mathcal O(h^\text{order})$ where $h$ is the spacing. It is required that `order > 1`.

```cpp
class bvp_k : public bvp
```
We initialize the solver:
```cpp
bvp_k::bvp_k(unsigned int order = 4, double tolerance = 1e-5);
```
where `order` is the order of the interpolating polynomial - 1, and `tolerance` initializes `tol`.

We solve a boundary value problem with:
```cpp
void bvp_k::ode_solve(
    arma::vec& x,
    arma::mat& U,
    const std::function<arma::rowvec(double,const arma::rowvec&)>& odefun,
    const std::function<arma::vec(const arma::rowvec&, const arma::rowvec&)>& bc
    );

void bvp_k::ode_solve(
    arma::vec& x,
    arma::mat& U,
    const std::function<arma::rowvec(double,const arma::rowvec&)>& odefun,
    const std::function<arma::mat(double, const arma::rowvec&)>& jacobian,
    const std::function<arma::vec(const arma::rowvec&, const arma::rowvec&)>& bc
    );
```
We are solving $u'=f(x,u)\land g(u(L),u(R)) = 0$, so `odefun` corresponds to $f$, and `bc` corresponds to $g$. The function `jacobian` corresponds to the derivative of $f$ with respect to $u$, i.e. $\frac{\partial f}{\partial u}$. It is necessary that `x` is initialized to a grid of points (of length $n$) with `x[0]`$=L$ and `x[n-1]`$=R$. The grid `x` should be sorted. The solver will solve the BVP at these grid points, so make sure this grid is dense. It is recommended that `U` is initialized to a guess of the initial problem (purhaps as outlined above), but this is not necessary. Regardless, the matrix must be initialized, this may achieved simply by setting `U = zeros(n,dim)` where `dim` is the dimension of the system of ODEs. The solution will overwrite `U`.

### Chebyshev Spectral Method
This method uses a spectrally converging method, where the BVP is solved at Chebyshev nodes scaled to the interval. Because the points are specific, an initial guess must be provided as a function that may be evaluated.

```cpp
class bvp_cheb : public bvp
```
We initialize the solver:
```cpp
bvp_cheb::bvp_cheb(unsigned int num_points = 32, double tolerance = 1e-5);
```
Where `num_points` is the number of points to use in the approximation, and `tolerance` initializes `tol`. Note that for problems with analytic solutions, `bvp_cheb` converges spectrally, thus only few points are ever needed (e.g. <50 points, while `bvp_k` may require >1000 points for a sufficiently accurate approximation).

We solve a boundary value problem with:
```cpp
void ode_solve(
    arma::vec& x,
    arma::mat& U,
    const std::function<arma::rowvec(double,const arma::rowvec&)>& odefun,
    const std::function<arma::vec(const arma::rowvec&, const arma::rowvec&)>& bc,
    const std::function<arma::rowvec(double)>& guess
    );

void ode_solve(
    arma::vec& x,
    arma::mat& U,
    const std::function<arma::rowvec(double,const arma::rowvec&)>& odefun,
    const std::function<arma::mat(double, const arma::rowvec&)>& jacobian,
    const std::function<arma::vec(const arma::rowvec&, const arma::rowvec&)>& bc,
    const std::function<arma::rowvec(double)>& guess
    );
```
Where `odefun`, `jacobian`, and `bc` are just as in `bvp_k`. With `bvp_cheb`, the parameter `x` should be initialized to a vector of length 2 and represent the boundaries of the interval, i.e. `x = {L,R}`. The matrix `U` may be empty. The parameter `guess` is an initial guess of $u(x)$, but may just return a row of zeros you have no guess. It is important that `guess(x).n_elem = dim` for all `x`$\in[L,R]$, where `dim` is the dimension of the system of ODEs. Both `x` and `U` will be set to the Chebyshev grid and approximate solution. This solution may be interpolated using a polynomial to get an approximation of the solution everywhere on the interval.

### Lobatto IIIa method
This method uses a 4th order Lobatto IIIa collocation formula.
```cpp
class bvpIIIa : public bvp
```
We initialize the solver:
```cpp
bvpIIIa::bvpIIIa(double tolerance = 1e-5);
```
where `tolerance` initializes `tol`.

We solve a boundary value problem with:
```cpp
void bvp_k::ode_solve(
    arma::vec& x,
    arma::mat& U,
    const std::function<arma::rowvec(double,const arma::rowvec&)>& odefun,
    const std::function<arma::vec(const arma::rowvec&, const arma::rowvec&)>& bc
    );

void bvp_k::ode_solve(
    arma::vec& x,
    arma::mat& U,
    const std::function<arma::rowvec(double,const arma::rowvec&)>& odefun,
    const std::function<arma::mat(double, const arma::rowvec&)>& jacobian,
    const std::function<arma::vec(const arma::rowvec&, const arma::rowvec&)>& bc
    );
```
Which is treated exactly in the same way as `bvp_k`. This method may be slower than `bvp_k` but is typically more stable (preferable for stiff problems).

### Poisson Solver
Given a rectangular region $\Omega$ in the $x,y$ plane, we can numerically solve the Poisson/Helmholtz equation $(\nabla^2 + k^2) u = f(x,y)$ with boundary conditions $u(x,y)=g(x,y)$ on $\partial\Omega$ using similar procedures to solving linear ODEs.

We solve the problem with the function:
```cpp
void poisson_helmholtz_2d(
    arma::mat& X,
    arma::mat& Y,
    arma::mat& U,
    const std::function<arma::mat(const arma::mat&, const arma::mat&)>& f,
    const std::function<arma::mat(const arma::mat&, const arma::mat&)>& bc,
    double eig = 0,
    int num_grid_points = 32);
```
We initialize `X` with the bounds in $x$, i.e. `X = {xLB, xUB}`, and the same for `Y`, i.e. `Y = {yLB, yUB}`. The matrix `U` does not need to be intialized. The function `bc` should equal `g(x,y)`. The parameter `eig` is $k$, which is `0` by default corresponding to Poisson's equation.

`X,Y,U` will be overwritten with the grid points solved for.

The parameter `num_grid_points` is the number of grid points along each axis, meaning the total number of points solved for will be `num_grid_points`^2. This solver uses the Chebyshev spectral order method only. The solver is take $\mathcal O(n^6)$ with respect to `num_grid_points`.