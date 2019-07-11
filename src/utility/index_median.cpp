#include <numerics.hpp>

uint numerics::index_median(const arma::vec& x) {
    std::vector<std::pair<double,uint>> y;
    for (uint i=0; i < x.n_elem; ++i) {
        y.push_back( {x(i),i} );
    }
    int nhalf = y.size()/2;
    std::nth_element(y.begin(), y.begin()+nhalf, y.end(), [](const std::pair<double,uint>& a, std::pair<double,uint>& b) -> bool {return a.first < b.first;});
    return y.at(nhalf).second;
}