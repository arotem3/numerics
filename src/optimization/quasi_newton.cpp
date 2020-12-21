#include <numerics.hpp>

void numerics::optimization::QausiNewton::_initialize(const arma::vec& x, const VecFunc& f, const MatFunc* jacobian)  {
    _F = f(x);
    if (jacobian == nullptr) _J = numerics::approx_jacobian(f,x);
    else _J = (*jacobian)(x);
}

void numerics::optimization::QausiNewton::_solve(arma::vec& x, const VecFunc& f, const MatFunc* jacobian) {
    numerics::optimization::VerboseTracker T(_max_iter);
    if (_v) T.header("max|f|");
    
    _initialize(x, f, jacobian);
    arma::vec dx, F1;

    _n_iter = 0;
    while (true) {
        bool successful_step = _step(dx, F1, x, f, jacobian);

        if (not successful_step) {
            _exit_flag = 3;
            if (_v) T.nan_flag();
            return;
        }
        double xtol = _xtol*std::max(1.0, arma::norm(x,"inf"));
        double df = arma::norm(F1 - _F, "inf");
        double ftol = _ftol*std::max(1.0, arma::norm(_F,"inf"));

        x += dx;
        _n_iter++;
        
        _F = std::move(F1);

        if (_v) T.iter(_n_iter, arma::norm(_F,"inf"));
        
        if (df < ftol) {
            _exit_flag = 0;
            if (_v) T.success_flag();
            return;
        }

        if (arma::norm(dx,"inf") < xtol) {
            _exit_flag = 1;
            if (_v) T.success_flag();
            return;
        }

        if (_n_iter >= _max_iter) {
            _exit_flag = 2;
            if (_v) T.max_iter_flag();
            return;
        }
    }
}