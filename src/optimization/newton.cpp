#include <numerics.hpp>

void numerics::optimization::Newton::fsolve(arma::vec& x, const VecFunc& f, const MatFunc& jacobian) {
    _check_loop_parameters();

    arma::vec dx;
    u_long k = 0;
    do {
        if (k >= _max_iter) {
            _exit_flag = 1;
            _n_iter += k;
            return;
        }
        _F = -f(x);
        _J = jacobian(x);
        if (_use_cgd) cgd(_J,_F,dx);
        else dx = arma::solve(_J,_F);
        x += dx;
        k++;
    } while ( arma::norm(dx, "inf") > _tol );
    _n_iter += k;
    _exit_flag = 0;
}