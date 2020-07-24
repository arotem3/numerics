#include <numerics.hpp>

void numerics::optimization::MomentumGD::minimize(arma::vec& x, const dFunc& f, const VecFunc& grad_f) {
    _check_loop_parameters();
    u_long k = 0;
    arma::vec y;
    if (not _line_min) y = x;
    
    while(true) {
        if (k >= _max_iter) {
            _exit_flag = 1;
            _n_iter += k;
            return;
        }
        _g = grad_f(x);

        if (_g.has_nan() || _g.has_inf()) {
            _exit_flag = 2;
            _n_iter += k;
            return;
        }

        if (arma::norm(_g,"inf") < _tol) {
            _n_iter += k;
            _exit_flag = 0;
            return;
        }

        if (_line_min) {
            _alpha = fminsearch([this,&x,&f](double a) -> double {return f(x - std::abs(a)*_g);}, _alpha);
            if (std::isnan(_alpha) || std::isinf(_alpha)) {
                _exit_flag = 2;
                _n_iter += k;
                return;
            }
            _alpha = std::abs(_alpha);
            x -= _alpha * _g;
        } else {
            arma::vec ym1 = y;
            y = x - _alpha * _g;
            x = y + _damping_param * (y - ym1);
            // if (k == 0) p = _g;
            // else p = _damping_param*p + _g;
            // x -= _alpha * p;
        }
        k++;
    }
}