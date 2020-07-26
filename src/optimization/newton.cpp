#include <numerics.hpp>

void numerics::optimization::Newton::fsolve(arma::vec& x, const VecFunc& f, const MatFunc& jacobian) {
    _check_loop_parameters();

    arma::vec dx;
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
        _F = -f(x);
        if (_v) T.iter(k, arma::norm(_F,"inf"));
        _J = jacobian(x);
        if (_use_cgd) cgd(_J,_F,dx);
        else dx = arma::solve(_J,_F);
        x += dx;
        k++;
    } while ( arma::norm(dx, "inf") > _tol );
    _n_iter += k;
    _exit_flag = 0;
    if (_v) T.success_flag();
}