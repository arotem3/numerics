#ifndef NUMERICS_INTEGRALS_HPP
#define NUMERICS_INTEGRALS_HPP

#include <unordered_map>
#include <queue>
#include <array>

namespace numerics
{

/* adaptive simpson's method, generally efficient.
 * --- fmap : all function evaluations will be stored here.
 * --- f : function to integrate.
 * --- a,b : interval [a,b] to evaluate integral over.
 * --- tol : error tolerance, i.e. stopping criterion */
template<typename Real, class Func, class Container>
Real simpson_integral(Container& fvals, const Func& f, Real a, Real b, Real tol = 1e-6) {
    if (tol <= 0) throw std::invalid_argument("simpson_integral() error: require tol (=" + std::to_string(tol) + ") > 0");
    if (b <= a) throw std::invalid_argument("simpson_integral() error: (" + std::to_string(a) + ", " + std::to_string(b) + ") does not define a valid interval");

    Real integral = 0;
    Real l, c, r, mid1, mid2, h;
    c = (a + b) / 2;
    l = (a + c) / 2;
    r = (c + b) / 2;

    std::queue<std::array<Real,5>> q; q.push({a,l,c,r,b});
    std::queue<Real> tq; tq.push(tol);
    while (not q.empty()) {
        for (Real val : q.front()) {
            if (fvals.count(val) == 0) fvals[val] = f(val);
        }
        auto& [a,l,c,r,b] = q.front();
        q.pop();
        tol = tq.front(); tq.pop();
        
        h = (b - a) / 2;
        Real s1 = (1.0/3) * (fvals[a] + 4*fvals[c] + fvals[b]) * h;
        Real s2 = (1.0/6) * (fvals[a] + 4*fvals[l] + 2*fvals[c] + 4*fvals[r] + fvals[b]) * h;
        if (std::abs(s2 - s1) < 15*tol) integral += s2;
        else {
            mid1 = (a + l)/2; mid2 = (l + c)/2;
            q.push({a,mid1,l,mid2,c});
            tq.push(tol/2);

            mid1 = (c + r)/2; mid2 = (r + b)/2;
            q.push({c,mid1,r,mid2,b});
            tq.push(tol/2);
        }
    }
    return integral;
}

/* adaptive simpson's method, generally efficient.
 * --- f : function to integrate.
 * --- a,b : interval [a,b] to evaluate integral over.
 * --- tol : error tolerance, i.e. stopping criterion */
template<typename Real, class Func>
Real simpson_integral(const Func& f, Real a, Real b, Real tol=1e-6) {
    std::unordered_map<Real, Real> fmap;
    return simpson_integral(fmap, f, a, b, tol);
}

template<typename Real, class Func>
inline Real lobatto4(const Func& f, Real a, Real b) {
    static const Real lobatto_4pt_nodes[] = {-1, -0.447213595499958, 0.447213595499958, 1};
    static const Real lobatto_4pt_weights[] = {0.166666666666667, 0.833333333333333, 0.833333333333333, 0.166666666666667};

    Real h = 0.5 * (b - a);
    Real c = 0.5 * (b + a);

    Real S = 0;
    for (int i=0; i < 4; ++i) {
        S += lobatto_4pt_weights[i] * f(c + h * lobatto_4pt_nodes[i]);
    }

    return S*h;
}

template<typename Real, class Func>
inline Real lobatto7(const Func& f, Real a, Real b) {
    static const Real lobatto_7pt_nodes[] = {-1, -0.468848793470714, -0.830223896278567, 0, 0.830223896278567, 0.468848793470714, 1};
    static const Real lobatto_7pt_weights[] = {0.047619047619048, 0.431745381209863, 0.276826047361566, 0.487619047619048, 0.276826047361566, 0.431745381209863, 0.047619047619048};
    
    Real h = 0.5 * (b - a);
    Real c = 0.5 * (b + a);

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
            double mid = 0.5*(a+b);
            q.push({a, mid, tol*0.5});
            q.push({mid, b, tol*0.5});
        }
    }
    return integral;
}

}

#endif