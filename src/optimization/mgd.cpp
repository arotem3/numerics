#include <numerics.hpp>

bool numerics::optimization::MomentumGD::_step(arma::vec& dx, arma::vec& g1, const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) {
    if (_line_min) {
        _alpha = numerics::optimization::fminsearch([this,&x,&f](double a)->double{return f(x - std::abs(a)*_g);}, _alpha);
        _alpha = std::abs(_alpha);
        dx = -_alpha*_g;
    } else {
        arma::vec tmp;
        if (_y.is_empty()) tmp = x;
        else tmp = _y;
        
        _y = x - _alpha*_g;
        dx = -_alpha*_g + _damping_param*(_y - tmp);
    }

    if (dx.has_nan()) return false;

    g1 = df(x + dx);
    return true;
}
