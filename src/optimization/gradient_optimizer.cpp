#include <numerics.hpp>

void numerics::optimization::GradientOptimizer::_initialize(const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) {
    _g = df(x);
}

void numerics::optimization::GradientOptimizer::_solve(arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) {
    VerboseTracker T(_max_iter);
    if (_v) T.header("f");
    
    _initialize(x, f, df, hessian);
    arma::vec dx, g1;

    _n_iter = 0;
    while (true) {
        bool successful_step = _step(dx, g1, x, f, df, hessian);

        if (not successful_step) {
            _exit_flag = 3;
            if (_v) T.nan_flag();
            return;
        }
        double xtol = _xtol*std::max(1.0, arma::norm(x,"inf"));
        x += dx;
        _n_iter++;

        _g = std::move(g1);

        if (_v) T.iter(_n_iter, f(x));

        if (arma::norm(_g,"inf") < _ftol) {
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