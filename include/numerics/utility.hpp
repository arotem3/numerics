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

template<typename eT>
class CycleQueue
{
protected:
    const u_long _max_elem;
    u_long _size;
    u_long _head;
    std::vector<eT> _data;

public:
    // const std::vector<eT>& data;
    CycleQueue(u_long maxsize) : _max_elem(maxsize), _size(0), _head(0)
    {
        if (_max_elem < 1)
            throw std::invalid_argument("CycleQueue error: require size > 0");
    }

    void push(const eT& x) {
        if (_size < _max_elem) {
            _data.push_back(x);
            _size++;
        } else {
            _data.at(_head) = x;
            _head = (_head + 1) % _size;
        }
    }

    void push(eT&& x) {
        if (_size < _max_elem) {
            _data.push_back(x);
            _size++;
        } else {
            _data.at(_head) = x;
            _head = (_head + 1) % _size;
        }
    }

    eT& at(u_long i) {
        if (i >= _size) {
            throw std::range_error("index (=" + std::to_string(i) + ") out of range of CycleQueue of size=" + std::to_string(_size) + ", with maximum size=" + std::to_string(_max_elem));
        }
        u_long ind = (i + _head) % _size;
        return _data.at(ind);
    }

    const eT& at(u_long i) const {
        if (i >= _size) {
            throw std::range_error("index (=" + std::to_string(i) + ") out of range of CycleQueue of size=" + std::to_string(_size) + ", with maximum size=" + std::to_string(_max_elem));
        }
        u_long ind = (i + _head) % _size;
        return _data.at(ind);
    }

    eT& back() {
        if (_size == 0) {
            throw std::range_error("cannot access back of empty queue.");
        }
        return _data.at(mod(_head - 1, _size));
    }

    const eT& back() const {
        if (_size == 0) {
            throw std::range_error("cannot access back of empty queue.");
        }
        return _data.at(mod(_head - 1, _size));
    }

    eT& front() {
        if (_size == 0) {
            throw std::range_error("cannot access back of empty queue.");
        }
        return _data.at(mod(_head-_size,_size));
    }

    const eT& front() const {
        if (_size == 0) {
            throw std::range_error("cannot access back of empty queue.");
        }
        return _data.at(mod(_head-_size,_size));
    }

    u_long size() const {
        return _size;
    }

    void clear() {
        _data.clear();
        _size = 0;
        _head = 0;
    }
};

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