# `numerics.hpp` Documentation

## Table of Contents
* [utitlity functions](#utility-methods)
    * [true modulus](#modulus-operation)
    * [meshgrid](#meshgrid)
    * [Polynomial Derivatives and Integrals](#Polynomial-Derivatives-and-Integrals)
* [integration](#integration)
* [derivatives](#discrete-derivatives)
    * [finite difference methods](#finite-differences)
    * [spectral methods](#spectral-derivatives)
* [linear root finding and optimization](#linear-root-finding-and-optimization)
    * [conjugate gradient method](#conjugate-gradient-method)
    * [convex contraint linear maximization](#convex-constraint-linear-maximization)
* [nonlinear root finding](#nonlinear-root-finding)
    * [Newton's method](#newton's-method)
    * [Broyden's method](#broyden's-method)
    * [Levenberg-Marquardt method](#Levenberg-Marquardt-Trust-Region/Damped-Least-Squares)
    * [fixed point iteration](#Fixed-Point-Iteration-with-Anderson-Mixing)
    * [fzero](#fzero)
    * [secant method](#secant)
    * [bisection](#bisection-method)
* [nonlinear optimization](#nonlinear-optimization)
    * [Newton's Method](#Newton's-Method-for-minimization)
    * [BFGS](#Broyden–Fletcher–Goldfarb–Shanno-algorithm)
    * [LBFGS](#Limited-Memory-BFGS)
    * [momentum Gradient Descent](#Momentum-Gradient-Descent)
    * [Stochastic Gradient Descent](#stochastic-gradient-descent)
    * [nonlinear conjugate gradient](#Nonlinear-Conjugate-Gradient-Descent)
    * [adjusted Gradient Descent](#adjusted-gradient-descent)
    * [genetic Algorithms](#Genetic-Maximization-Algorithm)
* [interpolation](#interpolation)
    * [cubic spline](#cubic-interpolation)
    * [polynomial](#polynomial-interpolation)
    * [nearest neighbor](#nearest-neighbor-interpolation)
    * [linear spline](#linear-interpolation)
    * [sinc spline](#sinc-interpolation)
* [Data Science](#data-science)
    * [k-fold train test split](#k-fold-train-test-split)
    * [k-means clustering](#k-means-clustering)
    * [splines](#splines)
    * [kernel smoothing](#kernel-smoothing)
    * [Regularized Linear Least Squares Regression](#Regularized-Linear-Least-Squares-Regression)
* [`ODE.hpp`](#`ODE.hpp`-Documentation)

all definitions are members of namespace `numerics`, all examples assume:
```cpp
using namespace std;
using namespace arma;
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

## Quadrature and Finite Differences

### Integration
```cpp
double integrate(const function<double(double)>& f,
                 double a, double b,
                 integrator i = LOBATTO, err = 1e-5);
```
We have multiple options for integrating a function $f:\mathbb{R} \rightarrow \mathbb{R}$ over a finite range. Primarily, we use `integrate()`. If $f$ is smooth, then the default integrator is ideal, otherwise, we should opt to use Simpson's method.
```cpp
double f(double x) return exp(-x*x); // e^(-x^2)
double lower_bound = 0;
double upper_bound = 3.14;

double I = integrate(f, lower_bound, upper_bound);

double I_simp = integrate(f, lower_bound, upper_bound,
                            integrator::SIMPSON, 1e-6);
```
Given a very smooth function (analytic), we can approximate its integral with few points using polynomial interpolation. Traditionally, polynomial interpolation takes $\mathcal O(n^3)$ time, but since we can choose the set of points to interpolate over, we can use Chebyshev nodes and integrate the function in $\mathcal O(n\log n)$ time using the fast fourier transform. The resulting approximation improves spectrally with $n$.
```cpp
double chebyshev_integral(const function<double(double)>& f,
                          double a, double b,
                          unsigned int num_f_evals = 25);
```
Where `num_f_evals` is the number of unique function evaluations to use, which is 25 by default. Increasing `num_f_evals` improves the accuracy, but very few are actually needed to achieve machine precision.

_**note:**_ If the function is not (atleast) continuous, the approximation may quickly become ill conditioned.

If we want to integrate an $n^{th}$ dimensional function within a box, then we can attempt Monte Carlo integration:
```cpp
double mcIntegrate(const function<double(const arma::vec& x)>& f,
                   const arma::vec& lower_bound,
                   const arma::vec& upper_bound,
                   double err = 1e-2,
                   int N = 1e3);
```
This method adaptively sample points from the function, in regions of high variance and uses inference techniques to approximate a bound on the the error.

**note:** to get high precision estimates, the method requires large N, which can be slow, maybe slower than grid style integration.

**note:** In retrospect this method is not especially well designed, I might come back to it.

Example:

```cpp
double circle(const vec& x) {
    if (norm(x) <= 1) return 1;
    else return 0;
}
vec lower_bound = {-1,-1}; // -1 <= x,y <= 1
vec upper_bound = {1,1};

double area = mcIntegrate(circle, lower_bound, upperbound);

double err = 1e-4;
int sample_points = 1e4;
double better_estimate = mcIntegrate(circle, lower_bound, upper_bound, err, sample_points);
```

## Discrete Derivatives
### Finite Differences
```cpp
double deriv(const function<double(double)>& f, double x,
            double err = 1e-5, bool catch_zero = true);
```
This function uses 4th order finite differences and adaptively determines the derivative at a point `x` within an error bound (1e-5 by default). We can also ask the derivative function to catch zeros, i.e. round to zero whenever `|x| < err`; this option can make the approximation more efficient.

**note:** `deriv()` may never actually evaluate the function at the point of interest, which is ideal if the function is not well behaved there.

Example:
```cpp
double f(double x) return sin(x);

double x = 0;

double df = deriv(f,x); // should return 1.0

double g(double x) return cos(x);

double dg = deriv(g,x,1e-5,false); // should return 0.0 
/*
in this case we are better off if catch_zero = true because we would require only 4 function evals rather than 8 which would be required to verify the derivative truly equals 0.
*/
```
We can also approximate gradients and Jacobian matrices:
```cpp
arma::vec grad(const function<double(const arma::vec&)> f,
               const arma::vec& x,
               double err = 1e-5,
               bool catch_zeros = true);

void approx_jacobian(const function<arma::vec(const arma::vec&)> f,
                     arma::mat& J,
                     const arma::vec& x,
                     double err = 1e-2, bool catch_zero = true);

arma::vec jacobian_diag(const function<arma::vec(const arma::vec&)> f,
                        const arma::vec& x);
```
These functions are wrappers for the `deriv()` function. So the functionality is similar. The function `jacobian_diag` computes only the diagonal of the jacobian matrix.

**note:** `approx_jacobian()` has no output and requires a matrix input, this matrix is overwritten with the jacobian matrix and serves as the output.

Example:
```cpp
double f(const vec& x) return dot(x,x); // return x^2 + y^2 + ...

vec F(const vec& x) return x%(x+1); // returns x.*(x+1)

arma::vec x = {0,1,2};

vec gradient = grad(f,x); // should return [2*0, 2*1, 2*2] = [0,2,4]

mat Jac;
approx_jacobian(F,Jac,x);
/* Jac = [1.0   0.0   0.0
          0.0   3.0   0.0
          0.0   0.0   4.0];
*/
```

### Spectral Derivatives
Given function defined over an interval, we can approximate the derivative of the function with spectral accuracy using the `FFT`. The function `spectral_deriv()` does this:
```cpp
polyInterp spectral_deriv(const function<double(double)>& f,
                        double a, double b,
                        unsigned int sample_points = 50);
```
We sample the function at chebyshev nodes scaled to the interval [a,b]; the function returns a polynomial object that can be evaluated anywhere on the interval. We can specify more sample points for more accuracy.

Example:
```cpp
double f(double x) return sin(x*x);

auto df = spectral_deriv(f, -2, 2);

vec x = linspace(-2,2);
vec derivative = df(x);
```

## Linear Root Finding and Optimization
### Conjugate Gradient Method
Armadillo features a very robust `solve()` and `spsolve()` direct solvers for linear systems, but in the case of very large systems (especially sparse systems) iterative solvers may be more efficient. The functions `cgd()` and `sp_cgd()` solve the general systems of linear equations $A \mathbf{x}=\mathbf{b}$ when $A$ is symmetric positive definite or in the least squares sense ($A^TA\mathbf{x}=A^T\mathbf{b}$) otherwise by conjugate gradient method.
```cpp
void cgd(arma::mat& A, arma::mat& b, arma::mat& x, cg_opts&);
cg_opts cgd(arma::mat& A, arma::mat& b, arma::mat& x);

void sp_cgd(const arma::sp_mat& A, const arma::mat& b, arma::mat& x, cg_opts&);
cg_opts sp_cgd(const arma::sp_mat& A, const arma::mat& b, arma::mat& x);
```
where `cg_opts` is a `struct` used to pass parameters to the solver:
```cpp
struct cg_opts {
    unsigned int max_iter; // maximum number of iterations the solver is allowed to perform
    arma::mat preconditioner;// an invertible preconditioner matrix
    function<arma::vec(const arma::vec&)> sp_precond; // for some preconditioner M, returns M.i() * x
    double err; // error tolerance initialized to 1e-6
    bool is_symmetric; // tell the solver not to check for symmetry. is true by default
    int num_iters_returned; // the number of iterations the solver actually needed.
}
```
**note:** when solving for `x` in `A*x = b`, if `A` is not square or symmetric, the solver will set `b=A.t()*b` and  `A = A.t()*A`, so the matrix and vector will both be modified outside the scope of the function. The resulting system is has worse conditioning, so using a preconditioner may be improve performance.

**note:** in the sparse case, if $A$ is not symmetric positive definite, the solver will quit.

### Adjusted Gradient Descent


### Convex Constraint Linear Maximization

For solving linear __*maximization*__ problems with linear constraints, we have the simplex algorithm that computes solutions using partial row reduction:
```cpp
double simplex(arma::mat& simplex_mat, arma::vec& x);
double simplex(const arma::rowvec& F,
               const arma::mat& RHS,
               const arma::vec& LHS,
               arma::vec& x);
```
In the case that the user knows how to define the simplex matrix, we have the first definition. The function returns the __*maximum*__ value within a convex polygon. The location of the maximum is stored in `x`.

Otherwise, the user can specify the linear function to __*maximize*__ `f(x)`, say, by providing a row vector `F` such that `f(x)=F*x`. The constraints take the form: `RHS*x <=LHS`. The function returns the __*maximum*__ value of `f(x)` that satisfies the constraints, and the location of the max is stored in `x`.

## Nonlinear Root Finding
unless otherwise specified, all of the following functions take the form:
```cpp
void solver(const function<arma::vec(const arma::vec&)>& f,
            arma::vec& x,
            options& opts);

options solver(const function<arma::vec(const arma::vec&)>& f,
            arma::vec& x);
```
### Option Stucts
All nonlinear solvers have option structs which all have the following variables:
```cpp
double err; // stopping tolerance
unsigned int max_iter; // max interations allowed
unsigned int num_iters_returned; // actual number of iterations needed, modified during function call
unsigned int num_FD_approx_needed; // number FD approximations computed, modified during function call
```
Jacobian/Hessian based solvers also ask for:
```cpp
bool use_FD_jacobian; // approximate jacobian by 4th order finite differences

arma::mat* init_jacobian; // provide an initial jacobian

arma::mat* init_jacobian_inv; // provide initial jacobian inverse

function<arma::mat(const arma::vec&)> jacobian_func; // provide jacobian function

arma::mat final_jacobian; // stores the last computed jacobian
```

### Newton's method
This is an implementation of Newton's method for systems of nonlinear equations. A jacobian function of the form above is required. As well as a good initial guess:
```cpp
void newton(const function<arma::vec(const arma::vec&)>& f,
            const function<arma::mat(const arma::vec&)>& J,
            arma::vec& x,
            nonlin_opts& opts);
```
the initial guess should be stored in `x` and `newton()` will asign the solution it finds to `x`.

There is also a single variable version:
```cpp
double newton(const function<double(double)>& f,
              const function<double(double)>& df,
              double x,
              double err = 1e-10);
```

### Broyden's Method
This solver is similar to Newton's method, but does not require knowledge of a Jacobian; instead, the solver takes rank 1 updates of the estimated Jacobian using the secant equations [(wikipedia)](https://en.wikipedia.org/wiki/Broyden%27s_method). Providing an initial jacobian or jacobian function does improve the process, but this solver requires far fewer Jacobian evaluations than Newton's method.
```cpp
void broyd(const function& f, vec& x, nonlin_opts& opts);
```

### Levenberg-Marquardt Trust Region/Damped Least Squares
This solver performs Newton like iterations, replacing the Jacobian with a damped least squares version [(wikipedia)](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm). It is recommended that a jacobian function be provided to the solver, otherwise the jacobian will be approximated via finite differences.
```cpp
void lmlsqr(const function& f, vec& x, lsqr_opts& opts);
```
in `lsqr_opts` you can choose the damping parameter via `damping_param`, and the damping scale with `damping_scale`. There is also the option to activate `use_scale_invariance` which weighs the damping paramter according to each equation in the system, but this can lead to errors (with singular systems) if used innapropriately.

### Fixed Point Iteration with Anderson Mixing
Solves problems of the form $x = g(x)$. It is possible to solve a subclass of these problems using fixed point iteration i.e. $x_{n+1} = g(x_n)$, but more generally we can solve these systems using Anderson Mixing: $x_{n+1} = \sum_{i=p}^n c_i g(x_i)$, where $1 \leq p \leq n$ and $\sum c_i = 0$.
```cpp
void mix_fpi(const function<arma::vec(const arma::vec&)> g,
             arma::vec& x,
             fpi_opts& opts);
```
We can specify to the solver how many previous iterations we want to remember i.e. we can specify $(n-p)$ using the `steps_to_remember` parameter in `fpi_opts`.

### fzero
Adaptively selects between secant method, and inverse interpolation to find a *simple* root of a single variable function over the interval `[a,b]`.
```cpp
double fzero(const function<double(double)>& f, double a, double b);
```

### Secant
Uses the secant as the approximation for the derivative used in Newton's method. Attempts to bracket solution for faster convergence, so providing an interval rather than two initial guesses is best.
```cpp
double secant(const function<double(double)>& f, double a, double b);
```

### Bisection method
Uses the bisection method to find the solution to a nonlinear equation within an interval.
```cpp
double bisect(const function<double(double)>& f,
              double a, double b, double err = 1e-8);
```

## Nonlinear Optimization

For nonlinear **_minimization_** we have a wrapper function that accepts a lot of information from the user
```cpp
double minimize_unc(const function<double(const arma::vec&)> &f,
                    arma::vec& x,
                    optim_opts& opts);
```

To `opts` we can specify anything you might specify to a nonlinear solvers, or an unconstrained minimizer. You can specify a choice of solver from which `LBFGS` is the default. The default arguments to `minimize_unc()` do not include gradient or hessian information, but it is recommended that at least a gradient function be provided via the options struct for improved efficiency. 

Alternatively, users can use certain solvers directly...

### Newton's Method for minimization
```cpp
void newton(const function<double(const arma::vec&)>& f,
            const function<arma::vec(const arma::vec&)>& df,
            const function<arma::mat(const arma::vec&)>& H,
            arma::vec& x,
            nonlin_opts& opts);
```
This time around `f` is the objective function, `df` is the gradient, and `H` is the Hessian. The parameter `x` should be initialized to a good guess of the optimum location; the location of the optimum will be stored in `x` when it is found. This method differs from Newton's method because it uses the strong Wolfe conditions at each step. Wolfe condition parameters `wolfe_c1`, `wolfe_c2` can be passed by the options struct. The parameter `wolfe_scale` is used in searching for the line min at each step.

### Broyden–Fletcher–Goldfarb–Shanno algorithm
Uses the BFGS algorithm for minimization using the strong Wolfe conditions. This method uses symmetric rank 1 updates using the secant equation with the further constraint that the Hessian remain symmetric positive definite.
```cpp
void bfgs(const function<double(const arma::vec&)>& f,
          const function<arma::vec(const arma::vec&)>& df,
          arma::vec& x,
          nonlin_opts& opts);
```
**note:** like Newton's method, `bfgs()` stores the Hessian in memory, and solves a linear system at each step, this may become inneficient in space and time when the problem is sufficiently large.

### Limited Memory BFGS
Uses the limited memory BFGS algorithm, which differs from BFGS by storing a limited number of previous values of `x` and `df(x)` rather than a full matrix. The number of steps can be specified by `steps_to_remember` in `lbfgs_opts`.

```cpp
void lbfgs(const function<double(const arma::vec&)>& f,
           const const function<arma::vec(const arma::vec&)>& df,
           arma::vec& x,
           lbfgs_opts& opts);
```

### Momentum Gradient Descent
Uses momentum gradient descent using adaptive line minimization. The damping parameter (explained in this [article](https://distill.pub/2017/momentum/)) can be specified in the options class.
```cpp
void mgd(const function<double(const arma::vec&)>& f,
         arma::vec& x,
         gd_opts& opts);
```

### Stochastic Gradient Descent
Uses random mini-batch gradient descent using adaptive line minimization. The batch size may be specified in the options class. If a damping parameter is specified in the options class, mini-batch momentum will be used.
```cpp
void sgd(const function<double(const arma::vec&)>& f,
         arma::vec& x,
         gd_opts& opts);
```

### Nonlinear Conjugate Gradient Descent
Uses the conjugate gradient algorithm for nonlinear objective functions with adaptive line minimization. Although this method uses only gradient information, it benefits from quasi-newton like super-linear convergence rates.
```cpp
void nlcgd(const function<double(const arma::vec&)>& f,
           arma::vec& x,
           nonlin_opts& opts);
```

### Adjusted Gradient Descent
This method was designed to approximate the conjugate to the gradient by storing only one previous* step. This method benefits from super-linear convergence rates similar to quasi-newton methods.
```cpp
void adj_gd(const function<double(const arma::vec&)>& f,
           arma::vec& x,
           nonlin_opts& opts);
```

**note:** I designed this method from experimentation, and it seems effective.

### Genetic Maximization Algorithm
This method uses a genetic algorithm for _**maximization**_. This method has a variety of parameters for updating the population of parameters to minimize with respect to:
* `population_size` : number of samples.
* `reproduction_rate` : parameter for geometric probability distribution of "reproducing agents". i.e. if `reproduction_rate` is close to 1, then only the most fit will reproduce and the algorithm will converge more quickly at the cost of possibly not optimal results (such a getting stuck at local maxima). If `reproduction_rate` is close to 0, then most members will be able to participate at the cost of slower convergence. default value = 0.5.
* `diversity_limit` : number of iterations after which we stop incentivising variance in the population. A lower value means quicker convergence at the cost possible not optimal results (such as getting stuck at local optima).
* `mutation_rate` : rate at which to introduce random perturbation to the population. Values close to 1 result in a population with higher variance resulting in slower convergence. Values close to 0 result in a population with higher variance resulting in faster convergence at the cost possible not optimal results (such as getting stuck at local optima).

We can use this method for both box constrained and unconstrained maximization. The box constrained version:
```cpp
double genOptim(const function<double(const arma::vec&)>& f,
                arma::vec& x,
                const arma::vec& xMin, const arma::vec& xMax,
                gen_opts& opts);
```
Where `x` is constrained such that `xMin <= x <= xMax`. No initial value is needed and the sample space is sampled uniformly.

The unconstrained version:
```cpp
double genOptim(const function<double(const arma::vec&)>& f,
                arma::vec& x,
                gen_opts& opts)
```
where an initial guess for `x` should be provided (by setting the value of `x` directly) and some `search_radius` should be provided in the options class (the `search_radius` serves as the intial standard deviation of the population). The `search_radius` default value = 1, but this value should really be determined by the based on the application.

**note:** the value returned is the maxima of the objective function.

### Binary Descision Maximization
Given a function that inputs a vector of binary values returns a double, we can attempt to find an maximimum of this function using a genetic algorithm.
```cpp
double boolOptim(std::function<double(const arma::uvec&)> f,
                 arma::uvec& x,
                 unsigned int input_size);
```
where `input_size` is the size of `x` (i.e. the number of descisions).

## Interpolation
### Cubic Interpolation
```cpp
class CubicInterp
```
Fits piecewise cubic polynomials to data. The fitting occurs on construction:
```cpp
CubicInterp::CubicInterp(const arma::vec& x, const arma::mat& Y);
```
We can save/load an interpolating object to a stream (such as a file stream):
```cpp
CubicInterp::save(ostream& out);
CubicInterp::load(istream& in);
```
We can also load a saved object on construction:
```cpp
CubicInterp::CubicInterp(istream& in);
```
Note, the data matrix will be stored to the stream as part of the object and can be recovered when the object is loaded using:
```cpp
arma::vec CubicInterp::data_X(); // independent values
arma::mat CubicInterp::data_Y(); // dependent values
```

We can predict based on the interpolation using the `predict` member function or the `()` operator:
```cpp
arma::mat CubicInterp::predict(const arma::vec&);
arma::mat CubicInterp::operator()(const arma::vec&);
```

### Polynomial Interpolation
Class wrapper for armadillo's `polyfit` and `polyval` specialized for interpolation.
```cpp
class polyInterp
```
We initialize the object:
```cpp
polyInterp::polyInterp(const arma::vec& x, const arma::mat& Y);
```
We can save/load an interpolating object to a stream (such as a file stream):
```cpp
polyInterp::save(ostream& out);
polyInterp::load(istream& in);
```
We can also load a saved object on construction:
```cpp
polyInterp::polyInterp(istream& in);
```
Note, the data matrix will be stored to the stream as part of the object and can be recovered when the object is loaded using:
```cpp
arma::vec polyInterp::data_X(); // independent values
arma::mat polyInterp::data_Y(); // dependent values
```

We can predict based on the interpolation using the `predict` member function or the `()` operator:
```cpp
arma::mat polyInterp::predict(const arma::vec&);
arma::mat polyInterp::operator()(const arma::vec&);
```

If there is only a need to fit and interpolate a data set once, we may find it more efficient ($\mathcal O(n^3)\rightarrow\mathcal O(n^2)$) and numerically stable to interpolate using Lagrange interpolation:
```cpp
arma::mat lagrangeInterp(const arma::vec& x,
                         const arma::mat& Y,
                         const arma::vec& xgrid,
                         bool normalize = false);
```
where `xgrid` is the set of values to interpolate over.

For high order polynomial interpolation, there is a likely hazard of exteme fluctuations in the values of polynomial (Runge's Phenomenon). wE can address this problem in `lagrangeInterp` by setting `normalize=true`. If the $i^{th}$ Lagrange interpolating polynomial is $L_i(x) = \prod_{j\neq i}\frac{x - x_j}{x_i - x_j}$, then the interpolant is of the form: $f(x) = \sum_i y_i L_i(x)$. When `normalize=true`, the interpolant is instead: $\hat f(x)=\sum_i y_iL_i(x)e^{-(x-x_i)^2/\nu}$, where $\nu = \text{range}(x)/n$. This normalization helps whenever $x$ is approximately uniform.

_**note:**_ When using `normalize=true`, remember that the resulting function is not a polynomial. 

### Nearest Neighbor Interpolation
We can do basic interpolation (on sorted data) using a single nearest neighbor. The result is a piecewise constant function:
```cpp
arma::mat nearestInterp(const arma::vec& x,
                        const arma::mat& Y,
                        const arma::vec& xgrid);
```

### Linear Interpolation
We can perform piecewise linear interpolation (on sorted data):
```cpp
arma::mat linearInterp(const arma::vec& x,
                       const arma::mat& Y,
                       const arma::vec& xgrid);
```

### Sinc interpolation
Given sorted _**uniformly spaced**_ points on an interval, we can interpolate the data using a linear combination of sinc functions $\text{sinc}(x) = \frac{\sin(\pi x)}{\pi x}$:
```cpp
arma::mat sincInterp(const arma::vec& x,
                     const arma::mat& Y,
                     const arma::vec& xgrid);
```

## Data Science
### K-Fold Train Test Split
When performing cross validation we may want to split the data into training and testing subsets. The `k_fold` procedure splits the data into `k` equal sized subsets of the data (randomly selected) for repeated training and testing.
```cpp
struct data_pair {
    arma::mat X; // independent variable subset
    arma::mat Y; // dependent variable subset
    arma::umat indices; // indices in original set corresponding to this subset
    arma::umat exclude_indices; // indices not included in this subset
};
typedef vector<data_pair> folds;

folds k_fold(const arma::mat& X,
             const arma::mat& Y,
             unsigned int k = 2,
             unsigned int dim = 0);
```
The parameter `dim` informs over which dimension of `X,Y` to split over, if `dim` is 0, then we split up the columns, if `dim` is 1, then we split up the rows.

### K-Means Clustering
```cpp
class kmeans
```
we can cluster unlabled data using the kmeans algorithm. On construction, we must specify our data and the number of clusters; our data is provided as an `arma::mat` where each row is a data entry.
```cpp
kmeans::kmeans(arma::mat& data, int num_clusters);
```
We can save and load a `kmeans` object to file using stream objects (note, the data matrix will be stored to the stream as part of the object and can be recovered when the object is loaded).
```cpp
kmeans::save(ostream& out);

kmeans::load(istream& in);
kmeans::kmeans(istream& in); // can be loaded in on construction
```

Once our `kmeans` object is constructed we can get information about which data went to which cluster with `getClusters()` and we can get the cluster means themselves using `getCentroids()`:
```cpp
arma::vec kmeans::getClusters() const; 
/*
the i^th elem corresponds to the i^th data point. The i^th element contains the cluster number which is an int ranging on [0,k-1]
*/

arma::mat kmeans::getCentroids() const;
/*
the i^th row is the i^th cluster where i is ranging on [0, k-1]
/*
```

We can test our clustering by predicting the cluster of data the `kmeans` was not trained on, we can use either the `()` operator or `predict()`:
```cpp
arma::rowvec kmeans::operator()(const arma::mat& X);
arma::rowvec kmeans::predict(const arma::mat& X);
/*
the i^th elem corresponds to the i^th data point. The i^th element contains the cluster number which is an int ranging on [0,k-1]
*/

int kmeans::operator()(const arma::rowvec& X);
int kmeans::predict(const arma::rowvec& X);
/*
output in [0,k-1]
*/
```

We can also ask `kmeans` to give us all the data from our original data belonging to a specific cluster. We can do this with the `[]` operator or `all_from_cluster()`:
```cpp
arma::mat operator[](int i);
arma::mat all_from_cluster(int i);
```

We can print a summary of our `kmeans` clustering to a stream:
```cpp
ostream& kmeans::summary(ostream& out = cout);
```
We can also retrieve basic documentation for the class using the `kmeans::help(ostream& out = cout)` member function.

### Splines
Fit radial basis splines to any dimensional data set. The construction is based on both multivariate polynomial terms and a radial basis kernel (polyharmonic). We regularize the fit by constraining frobenius norm of the hessian. Essentially we are solving the quadratic optimization problem: $\min_{c,d}(y - Kc - Pd)^T (y - Kc - Pd) + \lambda c^T K c$. The parameter $\lambda \geq 0$ can be provided during construction, or can be determined automatically by cross validation (when $\lambda=0$, the resulting function interpolates. When $\lambda\rightarrow\infty$ the resulting function tends toward the polynomial fit). Spline fitting is based on [ridge regression](https://en.wikipedia.org/wiki/Tikhonov_regularization) with regularization matrix $\Gamma_{i,j}=\phi_j''(x_i)$. Where $\phi$ is the radial basis function, and $\alpha$ is some proporionality constant.

```cpp
class splines
```
We construct the object via any of the following:
```cpp
splines::splines(int m);
splines::splines(double lambda = -1, int m = 1);

// construct and fit
splines::splines(const arma::mat& X,
                 const arma::mat& Y,
                 int m = 1);
splines::splines(const arma::mat& X,
                 const arma::mat& Y,
                 double lambda,
                 int m);
```
Where `m` is the degree of the polynomial, and it should be noted, that the size of the system grows exponentially with `m` and becomes extremely ill conditioned for large `m`. It is recomended that `1 <= m <= 3`. Where `m=1` (linear) minimizes bias.

The parameter `lambda` $=\lambda$. When `lambda = -1` or `nan` we determine its value by cross validation.

We can fit after construction using:
```cpp
splines& fit(const arma::mat& X, const arma::mat& Y);
arma::mat fit_predict(const arma::mat& X, const arma::mat& Y);
```
Where `fit_predict` is equivalent to calling `obj.fit(X,Y).predict(X)`.

We can predict based on our fit using the following functions:
```cpp
arma::mat splines::predict(const arma::mat& X);
arma::mat splines::operator()(const arma::mat& X);
```

We can extract a variety of additional information from our fit:
```cpp
arma::mat splines::data_X(); // independent variable data matrix
arma::mat splines::data_Y(); // dependent variable data matrix

arma::mat splines::rbf(const arma::mat&); // evaluate unscaled radial basis function
arma::mat splines::rbf_coef() const; // return radial basis coefficients

arma::mat splines::polyKern(const arma::mat&); // evaluate unscaled polynomial basis functions
arma::mat splines::poly_coef() const; // return polynomial basis coefficients

double splines::gcv_score() const; // return generalized cross validation score from fit (MSE scaled by eff_df)
double splines::eff_df() const; // return approximate effective degrees of freedom determined from sensitivities of the system
double splines::smoothing_param() const; // return lambda
```
We can save and load a splines object to a stream (file):
```cpp
void splines::save(ostream& out);

void splines::load(istream& in);
splines::splines(istream& in); // load on construction
```

### Kernel Smoothing
Kernel smoothing may be applied to quickly approximate a function at a point $x_0$ by weighted sum of samples within a bandwidth $\beta$ of $x_0$. The weights are determined by a symmetric kernel function $K(\cdot,\cdot)$ of which there are variety of options (we define $K(x,x_0)=f\left(\frac{||x-x_0||}{\beta}\right)$):

* `RBF`: $f(r) = \frac{1}{\sqrt{2\pi}}e^{-r^2/2}$
* `square`: $f(r) =0.5I_{r\leq 1}(r)$
* `triangle`: $f(r) = (1-r)I_{r\leq 1}(r)$
* `parabolic`: $f(r) = \frac{3}{4}(1-r^2)I_{r\leq 1}(r)$

Choosing $\beta$ may be difficult, so, by default, the bandwidth will be determined by k-fold crossvalidation.

```cpp
class kernel_smooth
```

We construct a smoothing object via:
```cpp
typedef enum KERNELS {RBF,square,triangle,parabolic} kernels;

kernel_smooth::kernel_smooth(double bdw=0, kernels k=RBF);

// construct and fit
kernel_smooth::kernel_smooth(const arma::vec& x,
                             const arma::vec& y,
                             double bdw=0,
                             kernels k=RBF);
```
where `bdw` = $\beta$. Whenever `bdw=0`, we use cross validation.

We fit the object:
```cpp
kernel_smooth& kernel_smooth::fit(const arma::vec& x,
                                  const arma::vec& y);

arma::vec kernel_smooth::fit_predict(const arma::vec& x,
                                     const arma::vec& y);
```
where `obj.fit_predict(x,y)` is equivalent to `obj.fit(x,y).predict(x)`.

We can predict based on our fit accroding:
```cpp
arma::mat kernel_smooth::predict(const arma::mat& xgrid);
arma::mat kernel_smooth::operator()(const arma::mat& xgrid);

arma::mat kernel_smooth::predict(double xval);
arma::mat kernel_smooth::operator()(double xval);
```
The functions, `predict` and the operator `()` are equivalent.

The object retains additional information from fitting, 
```cpp
arma::vec kernel_smooth::data_X(); // independent variable data matrix
arma::vec kernel_smooth::data_Y(); // dependent variable data matrix
double kernel_smooth::bandwidth() const; // return the bandwidth
double kernel_smooth::MSE() const; // return the MSE from fit
```

The kernel smoothing object may be saved to a stream (such as a file), and loaded from one:
```cpp
void kernel_smooth::save(std::ostream& out);

void kernel_smooth::load(std::istream& in);
kernel_smooth::kernel_smooth(std::istream& in); // load on construction
```

### Regularized Linear Least Squares Regression
When fitting a large basis set to data (often the case in non-parametric modeling), overfitting becomes a significant problem. To combat this problem we can regularize our parameters during the fit. Essentially we are solving the minimization problem: $\min_c||y - \Phi c||^2 + \lambda c^T R c$. Where $\lambda \geq 0$ is the regularization parameter which can be determined from cross validation. When $\lambda =0$, the fit tends toward high variance. When $\lambda\rightarrow\infty$, the fit tends toward high bias. Determining $\lambda$ may be achieved by cross validation. By default $R = I$, which is similar to [support vector regression](https://en.wikipedia.org/wiki/Support-vector_machine) whenever a radial basis set is used.

```cpp
class regularizer
```
With constructor:
```cpp
regularizer::regularizer(double lambda = nan);
regularizer::regularizer(const arma::mat& R, double lambda = nan);

// construct and fit
regularizer::regularizer(const arma::mat& X,
                         const arma::mat& Y,
                         double lambda = nan,
                         bool use_conj_grad = true);
regularizer::regularizer(const arma::mat& X,
                         const arma::mat& Y,
                         const arma::mat& R,
                         double lambda = nan,
                         bool use_conj_grad = true);
```
where `R` is the regularizer matrix, if one is not provided, `R` will be set to the identity matrix (i.e. $L^2$ regularization). The parameter `lambda` is the regularization parameter; when `lambda=nan`, its value will be determined by cross validation (recommended). The parameter `use_conj_grad` tells the object whether to solve the least squares problem directly or iteratively using conjugate gradient method.

We fit the object and return the coefficient matrix $c$ such that $\hat Y = X c$:
```cpp
arma::mat regularizer::fit(const arma::mat& X,
                              const arma::mat& Y,
                              bool use_conj_grad = true);
```
where `obj.fit_predict(x,y)` is equivalent to `obj.fit(x,y).predict()`.

Our object retains some extra information from fit:
```cpp
arma::mat regularizer::coef(); // returns the coefficient matrix

double regularizer::MSE() const; // returns the MSE from fit
double regularizer::eff_df() const; // returns the approximate effictive degrees of freedom in the system determined from the sensitivity of the system
double regularizer::regularizing_param() const; // returns the regularization parameter
arma::mat regularizer::regularizing_mat() const; // returns the regualarizing matrix
```

# `ODE.hpp` Documentation
## Table of Contents
* [`numerics.hpp` documentation](#`numerics.hpp-documentation`)
* [differential operators](#differential-operators)
* [Initial Value Problems](#Initial-Value-Problem-Solvers)
    * [general solver](#general-solver)
    * [Dormand-Prince 4/5](#dormand-prince-4/5)
    * [backward differentiation formula](#backward-differentiation-formula)
    * [Runge-Kutta $\mathcal O(4)$](#runge-kutta-fourth-order)
    * [Runge-Kutta $\mathcal O(5)$](#runge-kutta-fifth-order)
    * [backwards Euler](#backwards-euler)
    * [Adams-Moulton second order](#adams-moulton-second-order)
* [Boundary Value Problems](#boundary-value-problems-solver)
* [Poisson's Equation](#poisson-solver)

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

Given a linear ODE of the form: $y' + \alpha y = f(x)$ and the initial condition: $y(L) = \beta$, we can approximate the solution by solving the linear system: $(D+\alpha I)y = f(x) \land y(L) = \beta$. This can be solved either by concatentating the constraint to the rest of the system, or by replacing the first row of the system with the constraint. This might look like:
```cpp
vec f(vec& x) {
    // do something
}
mat D;
vec x, y;
double L, R, alpha, beta;
int N;
int method; // set to 1 or 2

diffmat2(D,x,L,R,N); // or diffmat4(), or cheb()

mat A;
vec F;
if (method == 1) {
    A = zeros(N+1,N);
    A.rows(0,N-1) = D + alpha*eye(N,N);
    A(N,0) = 1; // initial condition added at the end, matrix not square ==> less efficient solve
    F = zeros(N+1);
    F.rows(0,N-1) = f(x);
    F(N) = beta;
} else if (method == 2) {
    A = D + alpha*eye(N,N);
    A.row(0) *= 0;
    A(0,0) = 1; // initial condition replaces first row i.e. drops the derivative condition at y(L)
    F = f(x);
    F(0) = beta;
}
y = solve(A,F); // the solution
```
If we have a system of $m$ ODEs, we can solve both initial value problems and boundary value problems using a similar method where instead the operator is replaced with $(I_{m,m}\otimes D)$ and $f(x)$ is vectorized. Once a solution $y$ is found it is reshaped so that it is $n\times m$.

## Initial Value Problem Solvers
We define a system of initial value problem as having the form: $u' = f(t,u)$ with $u(0) = u_0$. Where $t$ is the independent variable and $u(t)$ is the dependent variable and is a row vector. All of the systems solvers are able to handle events. Some of the solvers have error control via adaptive step size selection. For the implicit solvers we can also provide a jacobian matrix $\frac{\partial f}{\partial u}$ to improve solver performance. All implicit solvers use Broyden's method to compute steps.

Solver interface:
```cpp
// system
typedef function<arma::rowvec(double,const arma::rowvec&)> odefun;

void solver(const odefun& f, arma::vec& t, arma::mat& u, ivp_options& opts);
ivp_options solver(const odefun& f, arma::vec& t, arma::mat& u);

// single variable
arma::vec solver(std::function<double(double,double)> f, arma::vec& t, double u0, ivp_options& opts);
arma::vec solver(std::function<double(double,double)> f, arma::vec& t, double u0);
```
The parameter `t` should be initialized to `{t_initial, t_final}`, this parameter will be overwritten with the grid points selected by the solver on the interval. For systems the parameter `u` should be initialized to $u_0$ as a single row vector. This parameter will be overwritten by the solver so that $u = u(t)$.

### General Solver
Returns a cubic interpolant object constructed from the solver output:
```cpp
typedef enum ODE_SOLVER {RK45,BDF23,RK4,RK5I,AM1,AM2} ode_solver;

CubicInterp ivp(f,t,u,opts, ode_solver s = RK45);
```
A choice of solver may be specified in addition to the other solver paramters. The solvers are explained below.

### Dormand-Prince 4/5
Fourth order explicit Runge-Kutta solver with adaptive step size for error control.
```cpp
void rk45(f,t,u,opts);
```

### Backward Differentiation Formula
Second order implicit linear multistep solver using the TR-BDF method to control error.
```cpp
void bdf23(f,t,u,opts);
```
This solver accepts a jacobian matrix.

### Runge-Kutta Fourth Order
classical Fourth order explicit Runge-Kutta solver with constant step size.
```cpp
void rk4(f,t,u,opts);
```

### Runge-Kutta Fifth Order
Fifth order implicit Runge-Kutta solver with constant step size.
```cpp
void rk5i(f,t,u,opts);
```
This object accepts a jacobian matrix.

### Backwards Euler
First order implicit Euler's method with constant step size. (only practical for demonstrative purposes, this method is innaccurate):
```cpp
void am1(f,t,u,opts);
```

### Adams-Moulton Second Order
Second order implicit linear multistep method with constant step size.
```cpp
void am2(f,t,u,opts);
```
For a more extensive explaination of the functionality of any of these methods may be explored in the example file.

## Boundary Value Problems Solver
We can solve boundary value problems using finite difference methods. This method was described in the [operators section](#differentiation-operators), but this method is far more generalized. Our problem defined as follows:

Given interval domain $\Omega = [L,R]$, and _**system**_ of ODEs $u' = f(x,u)$ with boundary conditions $g(x)=0$ on $\partial\Omega$ which is equivalently defined: $g(u(L),u(R)) = 0)$. This general problem is solved (in the least squares sense) using Broyden's method which requires an initial guess $v(x)$ of the solution. One method for providing an initial guess is by solving the linearized problem $u' = \big(\frac{\partial f}{\partial u}\big|_{u=u_0}\big)\cdot u$ where $u_0$ should be either $u(L)$ or $u(R)$ that hopefully satisfies the boundary conditions.

For example, given interval $\Omega=[0,1]$. if one of the equations is $u' = u(1-u)$ with boundary condition $u(0)=0$, solve instead $u' = (1-2u\big|_{u=0})u = 1$. So the initial guess should be $v(x) = Ce^x$ where $C$ should be chosen so that it atleast approximately satisfies the other boundary conditions.
```cpp
typedef function<arma::rowvec(double, arma::rowvec&)> odefun;

struct bcfun {
    double xL;
    double xR;
    function<arma::rowvec(const arma::rowvec&, const arma::rowvec&)> func;
};

typedef function<arma::mat(const arma::vec&)> soln_init;

struct dsolnp {
    arma::vec independent_var_values;
    arma::mat solution_values;
    polyInterp soln;
};

dsolnp bvp(const odefun& f, const bcfun& g, const soln_init& v, bvp_opts& opts);
```
The output struct `dsolnp` has the solution `independent_var_values` $=x$, `solution_values` $=u(x)$, and `soln` is a polynomial interpolation object that is fitted if `opts.order = CHEBYSHEV`.

For a more extensive explaination of the functionality of `bvp` may be explored in the example file.

### Poisson Solver
Given a rectangular region $\Omega$ in the $x,y$ plane, we can numerically solve Poisson's equation $\nabla^2 u = f(x,y)$ with boundary conditions $u(x,y)=g(x,y)$ on $\partial\Omega$ using similar procedures to solving linear ODEs.
```cpp
typedef function<arma::vec(const arma::vec&, const arma::vec&)> pde2fun;

struct bcfun_2d {
    double lower_x, upper_x; // interval in x
    double lower_y, upper_y; // interval in y
    
    function<arma::vec(const arma::vec&)> lower_x_bc;
    // u(x=lower_x, y) = g(y)

    function<arma::vec(const arma::vec&)> upper_x_bc;
    // u(x=upper_x, y) = g(y)
    
    function<arma::vec(const arma::vec&)> lower_y_bc;
    // u(x, y=lower_y) = g(x)
    
    function<arma::vec(const arma::vec&)> upper_y_bc;
    // u(x, y=upper_y) = g(x)
};



struct soln_2d {
    arma::mat X, Y; // independent variables
    arma::mat U; // solution
    void save(ostream& out);
    void load(istream& in);
};

soln_2d poisson2d(const pde2fun& f, const bcfun_2d& g, unsigned int num_pts = 48);
```
This solver uses the spectral order method only.