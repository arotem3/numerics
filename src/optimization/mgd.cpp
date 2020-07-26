#include <numerics.hpp>

void numerics::optimization::MomentumGD::minimize(arma::vec& x, const dFunc& f, const VecFunc& grad_f) {
    _check_loop_parameters();
    u_long k = 0;
    arma::vec y;
    if (not _line_min) y = x;
    
    VerboseTracker T(_max_iter);
    if (_v) T.header();
    while(true) {
        if (k >= _max_iter) {
            _exit_flag = 1;
            _n_iter += k;
            if (_v) T.max_iter_flag();
            return;
        }
        _g = grad_f(x);

        if (_g.has_nan() || _g.has_inf()) {
            _exit_flag = 2;
            _n_iter += k;
            if (_v) T.nan_flag();
            return;
        }

        if (arma::norm(_g,"inf") < _tol) {
            _n_iter += k;
            _exit_flag = 0;
            if (_v) T.success_flag();
            return;
        }

        if (_v) T.iter(k, f(x));

        if (_line_min) {
            _alpha = fminsearch([this,&x,&f](double a) -> double {return f(x - std::abs(a)*_g);}, _alpha);
            if (std::isnan(_alpha) || std::isinf(_alpha)) {
                _exit_flag = 2;
                _n_iter += k;
                if (_v) T.nan_flag();
                return;
            }
            _alpha = std::abs(_alpha);
            x -= _alpha * _g;
        } else {
            arma::vec ym1 = y;
            y = x - _alpha * _g;
            x = y + _damping_param * (y - ym1);
        }
        k++;
    }
}