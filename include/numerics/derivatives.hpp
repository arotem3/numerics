#ifndef NUMERICS_DERIVATIVES_HPP
#define NUMERICS_DERIVATIVES_HPP

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

#include <cmath>
#include <complex>
#include <algorithm>
#include <string>
#include <concepts>
#include <stdexcept>

namespace numerics
{
/* jacobian_diag(f, x, h, catch_zero, npt) : computes only the diagonal of a system of nonlinear equations.
 * --- f : f(x) system to approximate jacobian of.
 * --- x : vector to evaluate jacobian at.
 * --- h : finite difference step size. method is O(h^npt)
 * --- catch_zero : rounds near zero elements to zero.
 * --- npt : number of points to use in FD, npt = 1 uses f(x) and f(x+h), npt=2,4 do not use f(x) but instead use points f(x +/- h), f(x +/- 2h). Since npt=2 and npt=1 require the same number of f evals, npt 2 is used for its better accuracy. */
template<class Vec, std::invocable<Vec> Func, typename Real = typename Vec::value_type>
Vec jacobian_diag(Func f, const Vec& x, Real h = 100*std::sqrt(std::numeric_limits<Real>::epsilon()), short npt=1)
{
    u_long m = x.size();
    Vec J(m);
    Vec y = x;
    for (u_long i=0; i < m; ++i) {
        auto ff = [&f,&x,&y,i](Real z) -> Real
        {
            y[i] = z;
            Real f_y = f(y)[i];
            y[i] = x[i];
            return f_y;
        };
        J[i] = deriv(ff, x[i], h, npt);
    }
    return J;
}

template <class vec, std::invocable<vec> Func, typename real = typename vec::value_type>
vec hessian_diag(Func f, const vec& x, real h = 100 * std::cbrt(std::numeric_limits<real>::epsilon()))
{
    u_long m = x.size();
    vec B(m);
    vec y = x;
    real f0 = f(x);
    for (u_long i=0; i < m; ++i)
    {
        B[i] = -2*f0;
        real C = h * std::max<real>(std::abs(x[i]), 1);
        y[i] -= C;
        B[i] += f(y);
        y[i] = x[i] + C;
        B[i] += f(y);
        B[i] /= C*C; // (f(x-h) - 2*f(x) + f(x+h))/h^2
        y[i] = x[i];
    }

    return B;
}

/* grad(f, x, h, catch_zero, npt) : computes the gradient of a function of multiple variables.
 * --- f  : f(x) whose gradient to approximate.
 * --- x  : vector to evaluate gradient at.
 * --- h  : finite difference step size. method is O(h^npt).
 * --- catch_zero: rounds near zero elements to zero.
 * --- npt : number of points to use in FD, npt = 1 uses f(x) and f(x+h), npt=2,4 do not use f(x) but instead use points f(x +/- h), f(x +/- 2h). npt=1 is the default since it uses n+1 calls to f; npt=2 uses 2*n calls to f, and npt=4 uses 4*n calls to f. */
template<class Vec, std::invocable<Vec> Func, typename Real = typename Vec::value_type>
Vec grad(Func f, const Vec& x, Real h=100*std::sqrt(std::numeric_limits<Real>::epsilon()), short npt=1)
{
    u_long n = x.size();
    Vec g(n);
    if (npt == 1) { // specialize this instance to minimize repeated calculations.
        std::fill(g.begin(), g.end(), -f(x));
        Vec y = x;
        for (u_long i=0; i < n; ++i) {
            Real e = h*std::max<Real>(std::abs(y[i]), 1.0f);
            y[i] += e;
            g[i] += f(y);
            g[i] /= e;
            y[i] = x[i];
        }
    } else {
        Vec y = x;
        for (u_long i=0; i < n; ++i) {
            auto ff = [&f,&x,&y,i](Real t) -> Real
            {
                y[i] = t;
                Real f_y = f(y);
                y[i] = x[i];
                return f_y;
            };
            g[i] = deriv(ff, x[i], h, npt);
        }
    }
    
    return g;
}

/* deriv(f, x, h, catch_zero, npt) : computes the approximate derivative of a function of a single variable.
 * --- f  : f(x) whose derivative to approximate.
 * --- x  : point to evaluate derivative.
 * --- h  : finite difference step size; method is O(h^npt).
 * --- abstol: round anyvalue less than abstol to zero.
 * --- npt : number of points to use in FD, npt = 1 uses f(x) and f(x+h), npt=2,4 do not use f(x) but instead use points f(x +/- h), f(x +/- 2h). Since npt=2 and npt=1 require the same number of f evals, npt 2 is used for its better accuracy. */
template<typename Real, std::invocable<Real> Func>
inline Real deriv(Func f, const Real& x, Real h=100*std::sqrt(std::numeric_limits<Real>::epsilon()), short npt=2) {
    Real df;
    Real e = h*std::max<Real>(std::abs(x), 1.0f);
    if (npt == 1)
        df = (f(x + e) - f(x))/e;
    else if (npt == 2)
        df = (f(x + e) - f(x - e))/(2*e);
    else if (npt == 4)
        df = (f(x - 2*e) - 8*f(x - e) + 8*f(x + e) - f(x + 2*e))/(12*e);
    else
        throw std::invalid_argument("deriv() error: only 1, 2, and 4 point FD derivatives supported (but npt = " + std::to_string(npt) + " was provided).");
    
    return df;
}

namespace __directional_grad
{
    #ifdef NUMERICS_WITH_ARMA
    template <typename Real>
    Real norm_impl(const arma::Mat<Real>& x)
    {
        return arma::norm(x,"inf");
    }
    #endif
    template <class vec, typename Real=typename vec::value_type>
    Real norm_impl(const vec& x)
    {
        Real norm_of_x = 0;
        for (u_long i=0; i < x.size(); ++i)
            norm_of_x += std::max<Real>(norm_of_x, std::abs(x[i]));
        return norm_of_x;
    }
}

/* directional_grad(f, x, h, catch_zero, npt) : approximates J*v for J = jacobian of f; i.e. the gradient of f in the direction of v. Uses vectorized arithmetic operations and a norm(x) function so cannot be applied to std::vector with out creating these operators and a norm function. This function is especially designed to work with armadillo types.
 * --- f  : f(x) whose derivative to approximate
 * --- x  : point to evaluate derivative at
 * --- v  : direction of derivative
 * --- abstol : rounds near zero elements to zero if less than abstol.
 * --- npt : number of points to use in FD. */
template<class Vec, std::invocable<Vec> Func, typename Real = typename Vec::value_type>
Vec directional_grad(Func f, const Vec& x, const Vec& v, Real h=100*std::sqrt(std::numeric_limits<Real>::epsilon()), short npt=2) {
    if (x.size() != v.size()) {
        throw std::invalid_argument(
            "directional_grad() error: cannot compute derivative of f at x in the direction v when x.size() (="
            + std::to_string(x.size()) + ") does not match v.size() (="
            + std::to_string(v.size()) + ")."
        );
    }

    Vec Jv(x.size());

    const Real C = h * std::max(Real(1.0f), __directional_grad::norm_impl(x)) / std::max<Real>(1, __directional_grad::norm_impl(v));

    if (npt == 1)
        Jv = (f(x + C*v) - f(x)) / C;
    else if (npt == 2)
        Jv = (f(x + C*v) - f(x - C*v)) / (2*C);
    else if (npt == 4)
        Jv = (f(x - 2*C*v) - 8*f(x - C*v) + 8*f(x + C*v) - f(x + 2*C*v)) / (12*C);
    else
        throw std::invalid_argument("directional_grad() error: only 1, 2, and 4 point FD derivatives supported (but npt = " + std::to_string(npt) + " was provided).");

    return Jv;
}

#ifdef NUMERICS_WITH_ARMA
/* approx_jacobian(f, x, h, catch_zero, npt) : computes the jacobian of a system of nonlinear equations.
 * --- f  : f(x) whose jacobian to approximate.
 * --- x  : vector to evaluate jacobian at.
 * --- h : finite difference step size. method is O(h^npt)
 * --- catch_zero: rounds near zero elements to zero.
 * --- npt : number of number of points to use in FD, npt = 1 uses f(x) and f(x+h), npt=2,4 do not use f(x) but instead use points f(x +/- h), f(x +/- 2h). npt=1 is the default since it uses n+1 calls to f; npt=2 uses 2*n calls to f, and npt=4 uses 4*n calls to f. */
template<typename Real, std::invocable<arma::Col<Real>> Func>
arma::Mat<Real> jacobian(Func f, const arma::Col<Real>& x, Real h=100*std::sqrt(std::numeric_limits<Real>::epsilon()), short npt=1) {
    u_long n = x.size(); // num variables -> num cols
    if (n < 1) throw std::invalid_argument("approx_jacobian() error: when computing the jacobian, require x.size() (=" + std::to_string(n) + ") >= 1.");

    arma::Mat<Real> J;
    if (npt == 1) {
        J = arma::repmat(-f(x), 1, n);
        arma::Col<Real> y = x;
        for (u_long i=0; i < n; ++i) {
            Real e = h * std::max(std::abs(y[i]), Real(1.0f));
            y[i] += e;
            J.col(i) += f(y);
            J.col(i) /= e;
            y[i] = x[i];
        }
    }
    else if (npt == 2) {
        arma::Col<Real> y = x;
        for (u_long i=0; i < n; ++i) {
            Real e = h * std::max(std::abs(y[i]), Real(1.0f));
            y[i] += e;
            arma::Col<Real> df = f(y);
            y[i] = x[i] - e;
            df -= f(y);
            y[i] = x[i];
            
            df /= 2*e;
            if (J.empty()) J.set_size(df.size(), n);
            J.col(i) = df;
        }
    }
    else if (npt == 4) {
        arma::Col<Real> y = x;
        for (u_long i=0; i < x.n_elem; ++i) {
            Real e = h * std::max(std::abs(y[i]), Real(1.0f));
            y[i] = x[i] + 2*e;
            arma::Col<Real> df = -f(y);
            y[i] = x[i] + e;
            df += 8*f(y);
            y[i] = x[i] - e;
            df -= 8*f(y);
            y[i] = x[i] - 2*e;
            df += f(y);
            y[i] = x[i];

            df /= 12*e;
            if (J.empty()) J.set_size(df.size(), n);
            J.col(i) = df;
        }
    }
    else {
        throw std::invalid_argument("approx_jacobian() error: only 1, 2, and 4 point derivatives supported (not " + std::to_string(npt) + ").");
    }

    return J;
}

#ifdef NUMERICS_INTERPOLATION_COLLOCPOLY_HPP
template <typename Real, std::invocable<Real> Func>
ChebInterp<Real> spectral_deriv(Func f, Real a, Real b, u_long sample_points = 32) {
    std::complex<Real> i(0,1); // i^2 = -1
    u_long N = sample_points - 1;

    arma::Col<Real> y = arma::cos( arma::regspace<arma::Col<Real>>(0,N)*M_PI/N );
    arma::Col<Real> v = y;
    v.for_each([&f,&b,&a](Real& u){u = f(0.5f*(u+1)*(b-a)+a);});
    
    arma::uvec ii = arma::regspace<arma::uvec>(0,N-1);
    
    arma::Col<Real> V = arma::join_cols(v, arma::reverse(v(arma::span(1,N-1))));
    V = arma::Real(arma::fft(V));
    
    arma::Col<std::complex<Real>> u(2*N);
    u(arma::span(0,N-1)) = i*arma::regspace<arma::Col<std::complex<Real>>>(0,N-1);
    u(N) = 0;
    u(arma::span(N+1,2*N-1)) = i*arma::regspace<arma::Col<std::complex<Real>>>(1.0-(double)N, -1);
    
    arma::Col<Real> W = arma::Real(arma::ifft(u%V));
    W.rows(1,N-1) = -W.rows(1,N-1) / arma::sqrt(1 - arma::square(y.rows(1,N-1)));
    W(0) = 0.5*N*V(N) + arma::accu(arma::square(ii) % V.rows(ii)) / N;
    arma::Col<Real> j = arma::ones<arma::Col<Real>>(N); j.rows(arma::regspace<arma::uvec>(1,2,N-1)) *= -1;
    W(N) = 0.5*std::pow(-1,N+1)*N*V(N) + arma::accu(j % arma::square(ii) % V.rows(ii)) / N;
    W = W.rows(0,N);
    W /= (b-a)/2;

    return ChebInterp<Real>(0.5*(y+1)*(b-a) + a, W);
}
#endif // NUMERICS_INTERPOLATION_POLYNOMIAL_HPP
#endif // NUMERICS_WITH_ARMA

}

#endif // NUMERICS_DERIVATIVES_HPP