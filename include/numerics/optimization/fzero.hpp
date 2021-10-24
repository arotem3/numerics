#ifndef NUMERICS_OPTIMIZATION_FZERO_HPP
#define NUMERICS_OPTIMIZATION_FZERO_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <limits>
#include <type_traits>

#include "numerics/derivatives.hpp"

namespace numerics
{
namespace optimization
{

template <typename real, class Func, class Deriv, typename = typename std::enable_if<std::is_invocable_r<real,Deriv,real>::value>::type>
real newton_1d(Func f, Deriv df, real x, real tol=1e-4, u_long max_iter=100)
{
    if (tol <= 0)
        throw std::invalid_argument("error bound should be strictly positive, but tol=" + std::to_string(tol));
    
    constexpr real eps = std::numeric_limits<real>::epsilon();
    u_long k = 0;
    real fx, s;
    do {
        if (k >= max_iter) { // too many iterations
            std::cerr << "newton_1d() failed: too many iterations needed to converge." << std::endl
                      << "returing current best estimate."
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "|f(x)| = " << std::abs(f(x)) << " > tolerance" << std::endl << std::endl;
            return x;
        }
        fx = f(x);
        s = - fx / (df(x) + eps);
        x += s;
        ++k;
    } while ((std::abs(fx) > tol) && (std::abs(s) > tol));
    return x;
}

template <typename real, class Func>
real newton_1d(Func f, real x, real tol=1e-4, u_long maxit=100)
{
    auto df = [&](real u) -> real {
        return deriv(f, u);
    };
    return newton_1d(std::forward<Func>(f), df, x, tol, maxit);
}

template <typename real, class Func>
real secant(Func f, real a, real b, real tol=1e-4, u_long max_iter=100)
{
    real fa = f(a), fb = f(b);
    int k = 2;
    if (std::abs(fa) < tol) return a;
    if (std::abs(fb) < tol) return b;

    while (true) {
        if (k >= max_iter) { // too many iterations
            std::cerr << "secant() error: could not converge within " << max_iter << " function evaluations." << std::endl
                      << "\treturing current best estimate."
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "|f(x)| = " << std::min(std::abs(fa), std::abs(fb)) << " > tolerance" << std::endl << std::endl;
            return (std::abs(fa) < std::abs(fb)) ? a : b;
        }
        real c;

        real rel = std::max<real>( std::max<real>(std::abs<real>(b), std::abs<real>(a)), 1.0f);
        if (std::abs(fa - fb) < std::numeric_limits<real>::epsilon()*rel)
            c = (a + b) / 2;
        else
            c = b - fb*(b-a)/(fb-fa);

        real fc = f(c); k++;

        if (std::abs(fc) < tol)
            return c;

        if (fa*fb < 0) {
            if (fb*fc < 0) {
                a = c; fa = fc;
            } else {
                b = c; fb = fc;
            }
        } else {
            if (std::abs(fa) < std::abs(fb)) {
                b = c; fb = fc;
            } else {
                a = c; fa = fc;
            }
        }

        if (std::abs(a - b) < 2*std::numeric_limits<real>::epsilon())
            return (a+b)/2;
    }
}

template <typename real, class Func>
real bisect(Func f, real a, real b, real tol=1e-4)
{
    if (tol <= 0)
        throw std::invalid_argument("bisect() error: require tol (=" + std::to_string(tol) + ") > 0.");
    
    real fa = f(a), fb = f(b);

    if (std::abs(fa) < tol)
        return a;
    else if (std::abs(fb) < tol)
        return b;

    if (fa*fb > 0) { // bracketing error
        std::stringstream err;
        err << "bisect() error: provided points do not bracket a simple root."
            << "f(" << a << ") = " << fa << "), and f(" << b << ") = " << fb;
        throw std::invalid_argument(err.str());
    }

    real c, fc;

    do {
        c = (a+b)/2;
        fc = f(c);

        if (std::abs(fc) < tol)
            break;

        if (fc*fa < 0) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    } while ((std::abs(fc) > tol) && (std::abs(b-a) > tol));

    return (a + b)/2;
}

template <typename real, class Func>
real fzero(Func f, real a, real b, real tol=1e-4)
{
    u_long max_iter = std::min<u_long>(std::pow(std::log2((b-a)/tol)+1,2), 100); // will nearly never happen

    real c, d, e, fa, fb, fc, m=0, s=0, p=0, q=0, r=0, t, eps = std::numeric_limits<real>::epsilon();
    int k=0;
    fa = f(a); k++;
    fb = f(b); k++;
    if (std::abs(fa) == 0) return a;
    if (std::abs(fb) == 0) return b;

    if (fa*fb > 0) {
        std::stringstream err;
        err << "fzero() error: provided points do not bracket a simple root."
            << "f(" << a << ") = " << fa << "), and f(" << b << ") = " << fb;
        throw std::invalid_argument(err.str());
    }
    
    c = a; fc = fa; d = b-a; e = d;

    while (true) {
        if (std::abs(fc) < std::abs(fb)) {
             a =  b;  b =  c;  c =  a;
            fa = fb; fb = fc; fc = fa;
        }
        m = (c-b)/2;
        t = 2*std::abs(b)*eps + tol;
        if ((std::abs(m) < t) || (fb == 0))
            break; // convergence criteria

        if (k >= max_iter) {
            std::cerr << "fzero() error: could not converge within " << max_iter << " function evaluations (the estimated neccessary ammount).\n"
                      << "returing current best estimate.\n"
                      << "!!!---not necessarily a good estimate---!!!\n"
                      << "|dx| = " << std::abs(m) << " > " << tol << "\n";
            break;
        }

        if ((std::abs(e) < t) || (std::abs(fa) < std::abs(fb))) { // bisection
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

}
}

#endif