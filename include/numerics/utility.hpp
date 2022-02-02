#ifndef NUMERICS_UTILITY_HPP
#define NUMERICS_UTILITY_HPP

#include <algorithm>
#include <vector>
#include <utility>
#include <random>
#include <chrono>
#include <exception>

namespace numerics {

inline int mod(int a, int b) {
    return (a%b + b)%b;
}

template <class Vec, typename Real=decltype(Vec::value_type)>
u_long index_median(const Vec& x) {
    typedef std::pair<Real,u_long> val_idx;
    std::vector<val_idx> y(x.size());
    for (u_long i=0; i < x.size(); ++i) {
        y[i] = std::make_pair(x[i], i);
    }

    u_long nhalf = y.size() / 2;
    std::nth_element(y.begin(), y.begin()+nhalf, y.end(), [](const val_idx& a, const val_idx& b) -> bool {return a.first < b.first;});
    return y.at(nhalf).second;
}

template <class Vec>
u_long sample_from(const Vec& pmf, u_long seed=std::chrono::system_clock::now().time_since_epoch().count()) {
    typedef typename Vec::value_type Real;
    
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<Real> distribution(0.0, 1.0);
    
    u_long n = pmf.size();
    u_long i;
    Real cmf = 0;
    Real rval = distribution(generator);
    for (i = 0; i < n; ++i) {
        if ((cmf < rval) and (rval <= cmf + pmf[i]))
            break;
        cmf += pmf[i];
    }

    return i;
}

}
#endif