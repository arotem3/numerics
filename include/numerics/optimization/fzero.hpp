#ifndef NUMERICS_OPTIMIZATION_FZERO_HPP
#define NUMERICS_OPTIMIZATION_FZERO_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <limits>
#include <type_traits>
#include <cmath>
#include <concepts>

#include "numerics/derivatives.hpp"
#include "numerics/concepts.hpp"

namespace numerics {
    namespace optimization {
        // solves f(x) == 0 for x given an initial guess for x. The function df(x) is
        // the derivative of f(x). The method implements the Newton-Raphson method. see:
        // https://en.wikipedia.org/wiki/Newton%27s_method
        template <scalar_field_type T, std::invocable<T> Func, std::invocable<T> Deriv>
        T newton_1d(Func f, Deriv df, T x, precision_t<T> tol=10*std::sqrt(std::numeric_limits<precision_t<T>>::epsilon()), u_long max_iter=100)
        {
            if (tol <= 0)
                throw std::invalid_argument("error bound should be strictly positive, but tol=" + std::to_string(tol));
            
            constexpr precision_t<T> eps = std::numeric_limits<precision_t<T>>::epsilon();
            for (u_long k=0; k < max_iter; ++k)
            {
                if (k >= max_iter)
                    break;

                T fx = f(x);

                if (std::abs(fx) < tol)
                    return x;

                T s = - fx / (df(x) + eps);

                if (not std::isfinite(std::abs(s)))
                    break;

                x += s;
            }

            T fx = f(x);
            if (std::abs(fx) > tol)
                std::cerr << "newton_1d() failed: too many iterations needed to converge." << std::endl
                        << "returing current best estimate." << std::endl
                        << "!!!---not necessarily a good estimate---!!!" << std::endl
                        << "|f(x)| = " << std::abs(fx) << " > tolerance" << std::endl << std::endl;

            return x;
        }

        // solves f(x) == 0 for x given an initial guess for x. The method implements
        // the Newton-Raphson method. The derivative of f is approximated using finite
        // differences. see: https://en.wikipedia.org/wiki/Newton%27s_method
        template <scalar_field_type T, std::invocable<T> Func>
        T newton_1d(Func f, T x, precision_t<T> tol=10*std::sqrt(std::numeric_limits<precision_t<T>>::epsilon()), u_long maxit=100)
        {
            auto df = [&](T u) -> T {
                return deriv(f, u);
            };
            return newton_1d(std::forward<Func>(f), df, x, tol, maxit);
        }

        // solves f(x) == 0 for x given two initial guesses a and b using the secant
        // method. see: https://en.wikipedia.org/wiki/Secant_method
        template <scalar_field_type T, std::invocable<T> Func>
        T secant(Func f, T a, T b, precision_t<T> tol=10*std::sqrt(std::numeric_limits<precision_t<T>>::epsilon()), u_long max_iter=100)
        {
            typedef precision_t<T> prec;
            T fa = f(a), fb = f(b);
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
                T c;

                prec rel = std::max<prec>( std::max<prec>(std::abs(b), std::abs(a)), 1.0f);
                if (std::abs(fa - fb) < std::numeric_limits<prec>::epsilon()*rel)
                    c = T(0.5f) * (a + b);
                else
                    c = b - fb*(b-a)/(fb-fa);

                T fc = f(c); k++;

                if (std::abs(fc) < tol)
                    return c;

                if constexpr(std::is_same_v<T,prec>) {
                    if (fa*fb < 0) { // root bracketing?
                        if (fb*fc < 0) {
                            a = c; fa = fc;
                        } else {
                            b = c; fb = fc;
                        }
                    } else if (std::abs(fa) < std::abs(fb)) {
                        b = c; fb = fc;
                    } else {
                        a = c; fa = fc;
                    }
                } else {
                    if (std::abs(fa) < std::abs(fb)) {
                        b = c; fb = fc;
                    } else {
                        a = c; fa = fc;
                    }
                }

                if (std::abs(a - b) < 2*std::numeric_limits<prec>::epsilon())
                    return T(0.5f)*(a+b);
            }
        }

        // solves f(x) == 0 for x in the interval [a, b] using the bisection method. The
        // interval should guarantee a root, that is f(a)*f(b) < 0. A benefit of this
        // method is its guaranteed convergence in a fixed number of iterations
        // proportional to the log2 of the tolerance. see:
        // https://en.wikipedia.org/wiki/Bisection_method
        template <std::floating_point real, std::invocable<real> Func>
        real bisect(Func f, real a, real b, real tol=10*std::sqrt(std::numeric_limits<real>::epsilon()))
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

        // solves f(x) == 0 for x in the interval [a, b] using Brent's method for root
        // finding. The interval should guarantee a root, that is f(a)*f(b) < 0. This
        // method has worst case iterations to convergence that is the square of the
        // iterations required for the bisection method (which is often still quite
        // modest), but will usually converge much faster.
        // see:
        // Brent, R. P. (1973). Algorithms for minimization without derivatives.
        // Englewood Cliffs: Prentice-Hall.
        template <std::floating_point real, std::invocable<real> Func>
        real fzero(Func f, real a, real b, real tol=10*std::sqrt(std::numeric_limits<real>::epsilon()))
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
    } // namespace optimization
} // namespace numerics

#endif