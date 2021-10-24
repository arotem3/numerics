#include <numerics.hpp>

void numerics::PolyFeatures::fit(const arma::mat& x) {
    _dim = x.n_cols;
    _scale.set_size(_dim);
    for (u_long i=0; i < _dim; ++i) {
        _scale(i) = arma::norm(x.col(i),"inf");
    }
    std::queue<std::vector<u_int>> Q;
    std::set<std::vector<u_int>> S;
    for (u_int i=0; i < x.n_cols; ++i) {
        std::vector<u_int> str = {i};
        Q.push(str);
    }
    while ( !Q.empty() ) {
        std::vector<u_int> str = Q.front();
        if (str.size() > _deg) { // invalid monomial
            Q.pop();
            continue;
        }
        if (S.count(str) > 0) { // discovered
            Q.pop();
            continue;
        } else { // new node
            S.insert(str);
            for (uint i=0; i < x.n_cols; ++i) {
                std::vector<u_int> str2 = str;
                str2.push_back(i);
                std::sort( str2.begin(), str2.end() );
                Q.push(str2);
            }
            Q.pop();
        }
    }
    _monomials.clear();
    for (const std::vector<u_int>& str : S) _monomials.push_back(str);
}

arma::mat numerics::PolyFeatures::predict(const arma::mat& x) const {
    _check_x(x);
    u_int n = x.n_rows;
    u_int num_mons = _monomials.size();
    
    arma::mat P;
    if (_intercept) P = arma::ones(n, num_mons+1);
    else P = arma::ones(n, num_mons);

    u_int start = 0;
    if (_intercept) start = 1;
    for (u_int i=0; i < num_mons; ++i) {
        for (u_int r : _monomials.at(i) ) {
            P.col(i+start) %= x.col(r) / _scale(r);
        }
    }
    return P;
}

arma::mat numerics::PolyFeatures::fit_predict(const arma::mat& x) {
    fit(x);
    return predict(x);
};