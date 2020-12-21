#ifndef NUMERICS_UTILITY_HPP
#define NUMERICS_UTILITY_HPP

// --- utitility -------------- //
namespace constants {
    // --- integral constants
    const long double lobatto_4pt_nodes[4] = {-1, -0.447213595499958, 0.447213595499958, 1};
    const long double lobatto_4pt_weights[4] = {0.166666666666667, 0.833333333333333, 0.833333333333333, 0.166666666666667};

    const long double lobatto_7pt_nodes[7] = {-1, -0.468848793470714, -0.830223896278567, 0, 0.830223896278567, 0.468848793470714, 1};
    const long double lobatto_7pt_weights[7] = {0.047619047619048, 0.431745381209863, 0.276826047361566, 0.487619047619048, 0.276826047361566, 0.431745381209863, 0.047619047619048};
}

inline int mod(int a, int b) {
    return (a%b + b)%b;
}

template<typename eT> class CycleQueue {
    protected:
    u_long _max_elem;
    u_long _size;
    u_long _head;
    std::vector<eT> _data;

    public:
    // const std::vector<eT>& data;
    explicit CycleQueue(u_long size) /* : data(_data) */ {
        if (size < 1) throw std::runtime_error("cannot initialize CycleQueue to empty size");
        _max_elem = size;
        _size = 0;
        _head = 0;
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

void meshgrid(arma::mat&, arma::mat&, const arma::vec&, const arma::vec&);
void meshgrid(arma::mat&, const arma::vec&);

arma::uvec sample_from(int, const arma::vec&, const arma::uvec& labels = arma::uvec());
int sample_from(const arma::vec&, const arma::uvec& labels = arma::uvec());

uint index_median(const arma::vec& x);

#endif