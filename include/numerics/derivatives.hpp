#ifndef NUMERICS_DERIVATIVES_HPP
#define NUMERICS_DERIVATIVES_HPP

/* jacobian_diag(f, x, h, catch_zero, npt) : computes only the diagonal of a system of nonlinear equations.
 * --- f : f(x) system to approximate jacobian of.
 * --- x : vector to evaluate jacobian at.
 * --- h : finite difference step size. method is O(h^npt)
 * --- catch_zero : rounds near zero elements to zero.
 * --- npt : number of points to use in FD, npt = 1 uses f(x) and f(x+h), npt=2,4 do not use f(x) but instead use points f(x +/- h), f(x +/- 2h). Since npt=2 and npt=1 require the same number of f evals, npt 2 is used for its better accuracy. */
arma::vec jacobian_diag(const std::function<arma::vec(const arma::vec&)>& f, const arma::vec& x, double h = 1e-5, bool catch_zero=true, short npt=2);

/* approx_jacobian(f, x, h, catch_zero, npt) : computes the jacobian of a system of nonlinear equations.
 * --- f  : f(x) whose jacobian to approximate.
 * --- x  : vector to evaluate jacobian at.
 * --- h : finite difference step size. method is O(h^npt)
 * --- catch_zero: rounds near zero elements to zero.
 * --- npt : number of number of points to use in FD, npt = 1 uses f(x) and f(x+h), npt=2,4 do not use f(x) but instead use points f(x +/- h), f(x +/- 2h). npt=1 is the default since it uses n+1 calls to f; npt=2 uses 2*n calls to f, and npt=4 uses 4*n calls to f. */
arma::mat approx_jacobian(const std::function<arma::vec(const arma::vec&)>& f, const arma::vec& x, double h=1e-5, bool catch_zero = true, short npt=1);

/* grad(f, x, h, catch_zero, npt) : computes the gradient of a function of multiple variables.
 * --- f  : f(x) whose gradient to approximate.
 * --- x  : vector to evaluate gradient at.
 * --- h  : finite difference step size. method is O(h^npt).
 * --- catch_zero: rounds near zero elements to zero.
 * --- npt : number of points to use in FD, npt = 1 uses f(x) and f(x+h), npt=2,4 do not use f(x) but instead use points f(x +/- h), f(x +/- 2h). npt=1 is the default since it uses n+1 calls to f; npt=2 uses 2*n calls to f, and npt=4 uses 4*n calls to f. */
arma::vec grad(const std::function<double(const arma::vec&)>& f, const arma::vec& x, double h=1e-5, bool catch_zero = true, short npt=1);

/* deriv(f, x, h, catch_zero, npt) : computes the approximate derivative of a function of a single variable.
 * --- f  : f(x) whose derivative to approximate.
 * --- x  : point to evaluate derivative.
 * --- h  : finite difference step size; method is O(h^npt).
 * --- catch_zero: rounds near zero elements to zero.
 * --- npt : number of points to use in FD, npt = 1 uses f(x) and f(x+h), npt=2,4 do not use f(x) but instead use points f(x +/- h), f(x +/- 2h). Since npt=2 and npt=1 require the same number of f evals, npt 2 is used for its better accuracy. */
double deriv(const std::function<double(double)>& f, double x, double h=1e-5, bool catch_zero = true, short npt=2);

/* directional_grad(f, x, h, catch_zero, npt) : approximates J*v for J = jacobian of f; i.e. the gradient of f in the direction of v.
 * --- f  : f(x) whose derivative to approximate
 * --- x  : point to evaluate derivative at
 * --- v  : direction of derivative
 * --- catch_zero : rounds near zero elements to zero.
 * --- npt : number of points to use in FD. */
arma::vec directional_grad(const std::function<arma::vec(const arma::vec&)>& f, const arma::vec& x, const arma::vec& v, double h=1e-5, bool catch_zero=true, short npt=1);

Polynomial spectral_deriv(const std::function<double(double)>& f, double a, double b, uint sample_points = 50);

#endif