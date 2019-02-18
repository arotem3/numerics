# `numerics.hpp` Documentation

## Utility Methods

### Machine Epsilon
```cpp
double eps(double x = 1.0);
```


This function allows us to approximate the floating point value nearest to the floating point `x`, which is 1.0 by default.

Example:

```cpp
double x = 3.1415;
double machine_epsilon = eps(x);
```

### Mod operation
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
arma::mat meshgrid(arma::vec x);
```
This function constructs a 2D grid of points (potentially representing a region of the xy-plane), with points along one axis defined according to some vector `x`.

Example:

```cpp
arma::vec x = arma::regspace(0,2,0.1);
arma::mat XX = meshgrid(x); // X values
arma::mat YY = XX.t(); // Y values
```

## Quadrature and Finite Differences

### Integration
```cpp
double integrate(const function<double(double)>& f, double a, double b,
                        integrator i = LOBATTO, err = 1e-5);
```
We have multiple options for integrating a function $f:\mathbb{R} \rightarrow \mathbb{R}$ over a finite range. Primarily, we use `integrate()`. If $f$ is smooth, then the default integrator is ideal, otherwise, we can play around with with simpson's method or trapezoid rule.
```cpp

double f(double x) return exp(-x*x); // e^(-x^2)
double lower_bound = 0;
double upper_bound = 3.14;

double I = integrate(f, lower_bound, upper_bound);

double I_simp = integrate(f, lower_bound, upper_bound,
                            integrator::SIMPSON, 1e-10);

double I_trap = integrate(f, lower_bound, upper_bound,
                            integrator::TRAPEZOID, 1e-3);
```

If we want to integrate an $n^{th}$ dimensional function within a box, then we can attempt Monte Carlo integration:
```cpp
double mcIntegrate(const function<double(const arma::vec& x)>& f,
                   const arma::vec& lower_bound,
                   const arma::vec& upper_bound,
                   double err = 1e-2,
                   int N = 1e3);
```
This method adaptively sample points from the function, in regions of high variance and uses inference techniques to approximate a bound on the the error. 

**note:** to get high precision estimates, the method requires large N, which can be slow, maybe just as slow as grid style integration.

Example:

```cpp
double circle(const arma::vec& x) {
    if ( arma::norm(x) <= 1) return 1;
    else return 0;
}
arma::vec lower_bound = {-1,-1}; // -1 <= x,y <= 1
arma::vec upper_bound = {1,1};

double area = mcIntegrate(circle, lower_bound, upperbound);

double err = 1e-4;
int sample_points = 1e6;
double better_estimate = mcIntegrate(circle, lower_bound, upper_bound,
                                        err, sample_points);
```

### Discrete Derivatives
```cpp
double deriv(const function<double(double)>& f, double x,
            double err = 1e-5, bool catch_zero = true);
```
This function uses 4th order finite differences and adaptively determines the derivative at a point `x` within an error bound (1e-5 by default). We can also ask the derivative function to catch zeros, i.e. round to zero whenever `|x| < err`; this option can make the approximation more efficient.

**note:** `deriv()` may never actually evaluate the function at the point of interest, which is probably a good thing...

Example:
```cpp
double f(double x) return sin(x);

double x = 0;

double df = deriv(f,x); // should return 1.0

double g(double x) return cos(x);

double dg = deriv(g,x,1e-5,false); // should return 0.0 
/*
in this case we are better off if catch_zero = true because we would require only 4 function evals rather than 8.
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
```
These functions are wrappers for the `deriv()` function when our function. So the functionality is similar.

**note:** `approx_jacobian()` has no output and requires a matrix input, this matrix is overwritten with the jacobian matrix and serves as the output.

Example:
```cpp
double f(const arma::vec& x) return arma::norm(x); // return x^2 + y^2 + ...

arma::vec F(const arma::vec& x) return x%(x+1); // returns x.*(x+1)

arma::vec x = {0,1,2};

arma::vec gradient = grad(f,x); // should return [2*0, 2*1, 2*2] = [0,2,4]

arma::mat Jac;
approx_jacobian(F,Jac,x);
/* Jac = [1.0   0.0   0.0
          0.0   3.0   0.0
          0.0   0.0   4.0];
*/
```

Suppose we have a *periodic* function defined over an interval, then we can approximate the derivative of the function with spectral accuracy using the `FFT`. The function `spectral_deriv()` does this:
```cpp
arma::vec specral_deriv(const function<double(double)>& f,
                        arma::vec& interval,
                        int sample_points = 100);
```
We sample the function at discete uniformly spaced points; to emphasize: if `f` is periodic over `u(0) < x < u(1)` (i.e. $f(u_0) = f(u_1)$), then `spectral_deriv()` returns an accurate estimate of the derivative. We can specify more sample points for more accuracy.

Example:
```cpp
double f(double x) return sin(x*x); // periodic over any interval [-a,a]

arma::vec interval = {-2, 2};

double df = spectral_deriv(f, interval);
```

## Data Clustering
```cpp
class kmeans
```
we can cluster unlabled data using the kmeans algorithm. One construction, we must specify our data and the number of clusters; our data is provided as an `arma::mat` where each column is a data entry (this may change in the future).
```cpp
kmeans::kmeans(arma::mat& data, int num_clusters);
```
Since `kmeans` stores a pointer to the data, we require that the data exists through out the lifetime of the program.

We can save and load a `kmeans` object to file using stream objects.
```cpp
kmeans::save(ostream& out);

kmeans::load(istream& in);
kmeans::kmeans(istream& in); // can be loaded in on construction
```

Once our `kmeans` object is constructed we can get information about which data went to which cluster with `getClusters()` and we can get the cluster means themselves using `getCentroids()`:
```cpp
arma::rowvec kmeans::getClusters() const; 
/*
the i^th elem corresponds to the i^th data point. The i^th element contains the cluster number which is an int ranging on [0,k-1]
*/

arma::mat kmeans::getCentroids() const;
/*
the i^th column is the i^th cluster where i is ranging on [0, k-1]
/*
```

We can test our clustering by predicting the cluster of data the `kmeans` was not trained on, we can use either the `()` operator or `place_in_cluster()`:
```cpp
arma::rowvec kmeans::operator()(const arma::mat& X);
arma::rowvec kmeans::place_in_cluster(const arma::mat& X);
/*
the i^th elem corresponds to the i^th data point. The i^th element contains the cluster number which is an int ranging on [0,k-1]
*/

int kmeans::operator()(const arma::vec& X);
int kmeans::place_in_cluster(const arma::vec& X);
/*
output in [0,k-1]
*/
```

We can also ask `kmeans` to give us all the data from our original data belonging to a specific cluster. We can do this with the `[]` operator or `all_from_cluster()`:
```cpp
arma::mat operator[](int i);
arma::mat all_from_cluster(int i);
```

Finally, we can print a summary of our `kmeans` clustering to a stream:
```cpp
ostream& kmeans::summary(ostream& out);
```

## Linear Root Finding and Optimization

Armadillo features a very powerful `solve()` and `spsolve()` direct solvers for linear systems, but in the case of very large systems (especially sparse systems, iterative solvers may be more efficient). The functions `cgd()` and `sp_cgd()` solve the systems of linear equations `A*x=b` in the least squares sense using the conjugate gradient method.
```cpp
void cgd(arma::mat& A, arma::vec& b, arma::vec& x, cg_opts&);
cg_opts cgd(arma::mat& A, arma::vec& b, arma::vec& x);

void sp_cgd(const arma::sp_mat& A, const arma::vec& b, arma::vec& x, cg_opts&);
cg_opts sp_cgd(const arma::sp_mat& A, const arma::vec& b, arma::vec& x);
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
**note:** when solving for `x` in `A*x = b`, if `A` is not square or symmetric, the solver will set `b=A.t()*b` and  `A = A.t()*A`, so the matrix and vector will both be modified outside the scope of the function.

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

opts solver(const function<arma::vec(const arma::vec&)>& f,
            arma::vec& x);
```
### Option Stucts
All nonlinear solvers have option structs which all have the following variables:
```cpp
double err; // stopping tolerance
unsigned int max_iter; // max interations allowed
unsigned int num_iters_returned; // actual number of iterations needed
unsigned int num_FD_approx_needed; // number FD approximations computed
```
Jacobian/Hessian based solvers also ask for:
```cpp
bool use_FD_jacobian; // approximate jacobian

arma::mat* init_jacobian; // provide an initial jacobian

arma::mat* init_jacobian_inv; // initial jacobian inverse

function<arma::mat(const arma::vec&)> jacobian_func; // jacobian function

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
This solver is similar to Newton's method, but does not require knowledge of a Jacobian; instead, the solver takes rank 1 updates of the Jacobian using the secant equations [(wikipedia)](https://en.wikipedia.org/wiki/Broyden%27s_method). Providing an initial jacobian or jacobian function does improve the process, but this solver requires far fewer Jacobian evaluations than Newton's method.
```cpp
void broyd(const function& f, vec& x, nonlin_opts& opts);
```

### Levenberg-Marquardt Trust Region/Damped Least Squares
This solver performs Newton like iterations, replacing the Jacobian with a damped least squares version [(wikipedia)](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm). It is recommended that a jacobian function be provided to the solver.
```cpp
void lmlsqr(const function& f, vec& x, lsqr_opts& opts);
```
in `lsqr_opts` you can choose the damping parameter via `damping_param`, and the damping scale with `damping_scale`. There is also the option to activate `use_scale_invariance` which weighs the damping paramter according to each equation in the system, but this can lead to errors (with singular systems) if used innapropriately.

### Fixed Point Iteration with Anderson Mixing
Solves problems of the form $x = g(x)$. It is possible to solve asubclass of these problems using fixed point iteration i.e. $x_{n+1} = g(x_n)$, but more generally we can solve these systems using Anderson Mixing: $x_{n+1} = \sum_{i=p}^n c_i g(x_i)$, where $1 \leq p \leq n$ and $\sum c_i = 0$.
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

To opts we can specify anything you might specify to a nonlinear solvers, or an unconstrained minimizer. You can specify a choice of solver which is `LBFGS` by default. The default arguments to `minimize_unc()` do not include gradient or hessian information, but it is recommended that at least a gradient function be provided via the options struct for greater efficiency. 

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
...to be continued...