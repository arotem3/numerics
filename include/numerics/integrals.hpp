#ifndef NUMERICS_INTEGRALS_HPP
#define NUMERICS_INTEGRALS_HPP

#include <unordered_map>
#include <queue>
#include <array>
#include <string>
#include <stdexcept>

namespace numerics
{

/* adaptive simpson's method, generally efficient.
 * --- f : function to integrate.
 * --- a,b : interval [a,b] to evaluate integral over.
 * --- tol : error tolerance, i.e. stopping criterion */
template<typename Real, class Func>
Real simpson_integral(const Func& f, Real a, Real b, Real tol = 1e-6) {
    if (tol <= 0) throw std::invalid_argument("simpson_integral() error: require tol (=" + std::to_string(tol) + ") > 0");
    if (b <= a) throw std::invalid_argument("simpson_integral() error: (" + std::to_string(a) + ", " + std::to_string(b) + ") does not define a valid interval");

    Real integral = 0;
    Real m = (a + b) * 0.5f;
    Real fa = f(a), fb = f(b), fm = f(m);
    Real h = 0.5f * (b - a);
    Real s = (1.0f/3) * h * (fa + 4*fm + fb);

    std::queue<std::array<Real,8>> q; // a, c, b, fa, fc, fb, S, tol
    q.push({a, m, b, fa, fm, fb, s, tol});
    while (not q.empty()) {
        auto& [a, m, b, fa, fm, fb, s, tol] = q.front();
        q.pop();

        Real h = (b - a) * 0.5f;
        Real m1 = (a + m) * 0.5f, m2 = (m + b) * 0.5f;
        Real f1 = f(m1), f2 = f(m2);
        Real s1 = (1.0f/6) * h * (fa + 4*f1 + fm);
        Real s2 = (1.0f/6) * h * (fm + 4*f2 + fb);

        if (std::abs(s1 + s2 - s) < 15*tol)
            integral += s1 + s2;
        else {
            q.push({a, m1, m, fa, f1, fm, s1, tol*0.5f});
            q.push({m, m2, b, fm, f2, fb, s2, tol*0.5f});
        }
    }
    
    return integral;
}

// Gauss-Lobatto fixed quadrature of order 4
template<typename Real, class Func>
inline Real lobatto4(const Func& f, Real a, Real b) {
    static const Real lobatto_4pt_nodes[] = {-1, -0.447213595499958, 0.447213595499958, 1};
    static const Real lobatto_4pt_weights[] = {0.166666666666667, 0.833333333333333, 0.833333333333333, 0.166666666666667};

    Real h = 0.5f * (b - a);
    Real c = 0.5f * (b + a);

    Real S = 0;
    for (int i=0; i < 4; ++i) {
        S += lobatto_4pt_weights[i] * f(c + h * lobatto_4pt_nodes[i]);
    }

    return S*h;
}

// Gauss-Lobatto fixed quadrature of order 7
template<typename Real, class Func>
inline Real lobatto7(const Func& f, Real a, Real b) {
    static const Real lobatto_7pt_nodes[] = {-1, -0.468848793470714, -0.830223896278567, 0, 0.830223896278567, 0.468848793470714, 1};
    static const Real lobatto_7pt_weights[] = {0.047619047619048, 0.431745381209863, 0.276826047361566, 0.487619047619048, 0.276826047361566, 0.431745381209863, 0.047619047619048};

    Real h = 0.5f * (b - a);
    Real c = 0.5f * (b + a);

    Real S = 0;
    for (int i=0; i < 7; ++i) {
        S += lobatto_7pt_weights[i] * f(c + h * lobatto_7pt_nodes[i]);
    }

    return S*h;
}

/* adaptive gauss-Lobato's method, spectrally accurate for smooth functions
 * --- f : function to integrate.
 * --- a,b : interval [a,b] to evaluate integral over
 * --- tol : error tolerance, i.e. stopping criterion */
template<typename Real, class Func>
Real lobatto_integral(const Func& f, Real a, Real b, Real tol=1e-6) {
    if (tol <= 0) throw std::invalid_argument("lobatto_integral() error: require tol (=" + std::to_string(tol) + ") > 0");
    if (b <= a) throw std::invalid_argument("lobatto_integral() error: (" + std::to_string(a) + ", " + std::to_string(b) + ") does not define an interval");

    Real integral = 0;

    std::queue<std::array<Real, 3>> q; q.push({a,b,tol});

    while (not q.empty()) {
        auto& [a, b, tol] = q.front(); q.pop();

        Real sum4 = lobatto4(f, a, b);
        Real sum7 = lobatto7(f, a, b);
        if (std::abs(sum4 - sum7) < tol) integral += sum7;
        else {
            Real mid = 0.5f*(a+b);
            q.push({a, mid, tol*0.5f});
            q.push({mid, b, tol*0.5f});
        }
    }
    return integral;
}

}

#endif