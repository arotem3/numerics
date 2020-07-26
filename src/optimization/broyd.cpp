#include <numerics.hpp>


void numerics::optimization::Broyd::fsolve(arma::vec& x, const VecFunc& f) {
    _check_loop_parameters();
    arma::vec F1,dx,y;
    
    _F = f(x);
    _J = approx_jacobian(f,x);

    u_long k = 0;
    VerboseTracker T(_max_iter);
    if (_v) T.header("max|f|");
    do {
        if (k >= _max_iter) {
            _exit_flag = 1;
            _n_iter += k;
            if (_v) T.max_iter_flag();
            return;
        }

        if (_v) T.iter(k, arma::norm(_F,"inf"));

        dx = arma::solve(_J, -_F);
        x += dx;

        F1 = f(x);
        if (F1.has_nan() || F1.has_inf()) {
            _exit_flag = 2;
            _n_iter += k;
            if (_v) T.nan_flag();
            return;
        }

        y = (F1 - _F);
        _J += (y - _J*dx)*dx.t() / arma::dot(dx,dx);
        _F = F1;

        if (_F.has_nan() || _F.has_inf() || _J.has_nan() || _J.has_inf()) {
            _F = f(x);
            _J = approx_jacobian(f,x);
        }
        k++;
    } while (arma::norm(_F,"inf") > _tol);
    _n_iter += k;
    _exit_flag = 0;
    if (_v) T.success_flag();
}

void numerics::optimization::Broyd::fsolve(arma::vec& x, const VecFunc& f, const MatFunc& jacobian) {
    _check_loop_parameters();
    arma::vec F1,dx,y;
    
    _F = f(x);
    _J = jacobian(x);

    u_long k = 0;
    VerboseTracker T(_max_iter);
    if (_v) T.header("max|f|");
    do {
        if (k >= _max_iter) {
            _exit_flag = 1;
            _n_iter += k;
            if (_v) T.max_iter_flag();
            return;
        }

        if (_v) T.iter(k, arma::norm(_F,"inf"));

        dx = arma::solve(_J, -_F);
        x += dx;

        F1 = f(x);
        if (F1.has_nan() || F1.has_inf()) {
            _exit_flag = 2;
            _n_iter += k;
            if (_v) T.nan_flag();
            return;
        }

        y = (F1 - _F);
        _J += (y - _J*dx)*dx.t() / arma::dot(dx,dx);
        _F = F1;

        if (_F.has_nan() || _F.has_inf() || _J.has_nan() || _J.has_inf()) {
            _F = f(x);
            _J = jacobian(x);
        }
        k++;
    } while (arma::norm(_F,"inf") > _tol);
    _n_iter += k;
    _exit_flag = 0;
    if (_v) T.success_flag();
}