#ifndef NUMERICS_INTEGRALS_HPP
#define NUMERICS_INTEGRALS_HPP

#include <cmath>
#include <queue>
#include <array>
#include <string>
#include <stdexcept>
#include <iostream>

namespace numerics
{
    /* adaptive simpson's method, generally efficient.
    * --- f : function to integrate.
    * --- a,b : interval [a,b] to evaluate integral over.
    * --- tol : error tolerance, i.e. stopping criterion */
    template<typename Real, std::invocable<Real> Func>
    Real simpson(const Func& f, Real a, Real b, Real tol = std::sqrt(std::numeric_limits<Real>::epsilon())) {
        if (tol <= 0)
            throw std::invalid_argument("simpson() error: require tol (=" + std::to_string(tol) + ") > 0");
        if (b <= a)
            throw std::invalid_argument("simpson() error: (" + std::to_string(a) + ", " + std::to_string(b) + ") does not define a valid interval");

        Real integral = 0;
        Real m = (a + b) * 0.5f;
        Real fa = f(a), fb = f(b), fm = f(m);
        Real h = 0.5f * (b - a);
        Real s = (1.0f/3) * h * (fa + 4*fm + fb);
        Real kahan_c = 0; // for stable accumulation

        std::queue<std::array<Real,8>> q; // a, m, b, fa, fm, fb, S, tol
        q.push({a, m, b, fa, fm, fb, s, tol});

        bool failed=false;
        while (not q.empty()) {
            auto& [a, m, b, fa, fm, fb, s, tol] = q.front();
            q.pop();

            if ((1 + std::numeric_limits<Real>::epsilon())*a >= b) {
                if (not failed) {
                    std::cerr << "simpson() warning: could not improve quadrature approximation.\n";
                    failed = true;
                }
                Real y = (fa + fb) * (b - a) / 2 - kahan_c;
                Real t = integral + y;
                kahan_c = (t - integral) - y;
                integral = t;
                continue;
            }

            Real h = (b - a) * 0.5f;
            Real m1 = (a + m) * 0.5f, m2 = (m + b) * 0.5f;
            Real f1 = f(m1), f2 = f(m2);
            Real s1 = (1.0f/6) * h * (fa + 4*f1 + fm);
            Real s2 = (1.0f/6) * h * (fm + 4*f2 + fb);

            if (std::abs(s1 + s2 - s) < 15*tol) {
                Real y = (16*(s1 + s2) - s)/15 - kahan_c;
                Real t = integral + y;
                kahan_c = (t - integral) - y;
                integral = t;
            }
            else {
                q.push({a, m1, m, fa, f1, fm, s1, tol/2});
                q.push({m, m2, b, fm, f2, fb, s2, tol/2});
            }
        }
        
        return integral;
    }

    /* adaptive 4 point Gauss-Lobatto quadrature with 7 point Kronrod extension.
    * --- f : function to integrate.
    * --- a,b : interval [a,b] to evaluate integral over
    * --- tol : error tolerance, i.e. stopping criterion */
    template <typename Real, std::invocable<Real> Func>
    Real lobatto4k7(const Func& f, Real a, Real b, Real tol= std::sqrt(std::numeric_limits<Real>::epsilon()))
    {
        //   -1 ---   -z ---   -p ---    0 ---    p ---    z ---    1
        // w[1] ---    0 --- w[0] ---    0 --- w[0] ---    0 --- w[1]
        // k[3] --- k[2] --- k[1] --- k[0] --- k[1] --- k[2] --- k[3]
        //    a ---   x1 ---   x2 ---   m  ---   x3 ---   x4 ---    b
        static const Real p = 0.447213595499958;
        static const Real w4[] = {0.833333333333333, 0.166666666666667};
        static const Real z = 0.81649658092772603;
        static const Real k7[] = {0.4571428571428571, 0.4251700680272109, 0.2938775510204082, 0.05238095238095238};

        if (tol <= 0)
            throw std::invalid_argument("lobatto4k7() error: require tol (=" + std::to_string(tol) + ") > 0");
        if (b <= a)
            throw std::invalid_argument("lobatto4k7() error: (" + std::to_string(a) + ", " + std::to_string(b) + ") does not define an interval");

        Real integral = 0;
        Real kahan_c = 0;
        Real fa = f(a), fb = f(b);

        std::queue<std::array<Real,5>> q; // a, b, fa, fb, tol
        q.push({a,b,fa,fb,tol});

        bool failed = false;
        while (not q.empty())
        {
            auto& [a, b, fa, fb, tol] = q.front(); q.pop();

            if ((1 + std::numeric_limits<Real>::epsilon())*a >= b) {
                if (not failed) {
                    std::cerr << "lobatto4k7() warning: could not improve quadrature approximation.\n";
                    failed = true;
                }
                Real y = (fa + fb) * (b - a) / 2 - kahan_c;
                Real t = integral + y;
                kahan_c = (t - integral) - y;
                integral = t;
                continue;
            }

            Real h = (b - a)/2;
            Real m = (b + a)/2;
            Real x1 = m - h*z, x2 = m - h*p, x3 = m + h*p, x4 = m + h*z;
            Real fm = f(m), f1 = f(x1), f2 = f(x2), f3 = f(x3), f4 = f(x4);

            // lobatto 4
            Real sum4 = h * (w4[0]*(f2 + f3) + w4[1]*(fa + fb));

            // kronrod extension
            Real sum7 = h * (k7[0]*fm + k7[1]*(f2 + f3) + k7[2]*(f1 + f4) + k7[3]*(fa + fb));

            if (std::abs(sum4-sum7) < tol) {
                Real y = sum4 - kahan_c;
                Real t = integral + y;
                kahan_c = (t - integral) - y;
                integral = t;
            } else {
                Real t = tol * (1 - z)/2;
                q.push({a, x1, fa, f1, t});
                q.push({x4, b, f4, fb, t});
                t = tol * (z-p)/2;
                q.push({x1, x2, f1, f2, t});
                q.push({x3, x4, f3, f4, t});
                t = tol * p/2;
                q.push({x2, m, f2, fm, t});
                q.push({m, x3, fm, f3, t});
            }
        }

        return integral;
    }

    // adaptive 5 point Gauss-Lobatto quadrature embedded with 3 point Gauss-Lobatto
    // for error estimation, uses 5 point approximation. Whereas lobatto4k7 uses a
    // more accurate error estimate, it keeps the 4 point approximation (the kronrod
    // extension is not as accurate as the Gauss-Lobatto rule), instead lobatto5l3
    // keeps the 5 point approximation (because both the 5-point and 3-point are
    // quality quadrature rules).
    template <typename Real, std::invocable<Real> Func>
    Real lobatto5l3(const Func& f, Real a, Real b, Real tol= std::sqrt(std::numeric_limits<Real>::epsilon()))
    {
        static const Real p = 0.6546536707079771;
        static const Real w5[] = {0.7111111111111111, 0.5444444444444444, 0.10};

        if (tol <= 0)
            throw std::invalid_argument("lobatto5l3() error: require tol (=" + std::to_string(tol) + ") > 0");
        if (b <= a)
            throw std::invalid_argument("lobatto5l3() error: (" + std::to_string(a) + ", " + std::to_string(b) + ") does not define an interval");

        Real integral = 0;
        Real kahan_c = 0;
        Real fa = f(a), fb = f(b);

        std::queue<std::array<Real,5>> q;
        q.push({a,b,fa,fb,tol});

        bool failed=false;
        while (not q.empty())
        {
            auto& [a, b, fa, fb, tol] = q.front(); q.pop();

            if ((1 + std::numeric_limits<Real>::epsilon())*a >= b) {
                if (not failed) {
                    std::cerr << "lobatto5l3() warning: could not improve quadrature approximation.\n";
                    failed = true;
                }
                Real y = (fa + fb) * (b - a) / 2 - kahan_c;
                Real t = integral + y;
                kahan_c = (t - integral) - y;
                integral = t;
                continue;
            }

            Real m = (b + a) / 2;
            Real h = (b - a) / 2;

            Real fm = f(m);
            Real sum3 = (h/3) * (fa + 4*fm + fb);
            
            Real x1 = m - h*p, x2 = m + h*p;
            Real f1 = f(x1), f2 = f(x2);

            Real sum5 = h * (w5[0]*fm + w5[1]*(f1 + f2) + w5[2]*(fa + fb));

            if (std::abs(sum5 - sum3) < tol) {
                Real y = sum5 - kahan_c;
                Real t = integral + y;
                kahan_c = (t - integral) - y;
                integral = t;
            } else {
                Real t = tol * (1-p)/2;
                q.push({a, x1, fa, f1, t});
                q.push({x2, b, f2, fb, t});
                t = tol * p/2;
                q.push({x1, m, f1, fm, t});
                q.push({m, x2, fm, f2, t});
            }
        }
        return integral;
    }
} // namespace numerics

#endif