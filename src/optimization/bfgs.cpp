#include <numerics.hpp>

void numerics::optimization::BFGS::minimize(arma::vec& x, const dFunc& f, const VecFunc& grad_f) {
    _check_loop_parameters();
    u_long n = x.n_elem;
    arma::vec g1;
    _g = grad_f(x);
    if (_use_fd) {
        _H = arma::symmatu( approx_jacobian(grad_f,x) );
        bool chol_success = arma::inv_sympd(_H,_H);
        if (!chol_success) _H = arma::pinv(_H);
    } else _H = arma::eye(n,n);

    arma::vec p;
    double alpha;
    arma::vec s, y;
    u_long k = 0;
    do {
        if (k >= _max_iter) {
            _exit_flag = 1;
            _n_iter += k;
            return;
        }
        p = -(_H*_g);
        if (p.has_nan() || p.has_inf()) {
            if (_use_fd) {
                _H = arma::symmatu( approx_jacobian(grad_f,x) );
                bool chol_success = arma::inv_sympd(_H,_H);
                if (!chol_success) _H = arma::pinv(_H);
            } else _H = arma::eye(n,n);
            p = -(_H*_g);
        }
        alpha = wolfe_step(f,grad_f,x,p,_wolfe_c1,_wolfe_c2);
        s = alpha*p;

        if (s.has_nan() || s.has_inf()) {
            _n_iter += k;
            _exit_flag = 2;
            return;
        }

        x += s;
        g1 = grad_f(x);
        y = g1 - _g;

        double sdoty = arma::dot(s,y);
        arma::vec Hdoty = _H*y;
        _H += (1 + arma::dot(y,Hdoty) / sdoty) * (s*s.t())/sdoty - (s*Hdoty.t() + Hdoty*s.t())/sdoty;
        _g = g1;
        k++;
    } while (arma::norm(p,"inf") > _tol);
    _n_iter += k;
    _exit_flag = 0;
}

void numerics::optimization::BFGS::minimize(arma::vec& x, const dFunc& f, const VecFunc& grad_f, const MatFunc& hessian) {
    _check_loop_parameters();
    u_long n = x.n_elem;
    arma::vec g1;
    _g = grad_f(x);
    _H = hessian(x);
    bool chol_success = arma::inv_sympd(_H,_H);
    if (!chol_success) _H = arma::pinv(_H);

    arma::vec p;
    double alpha;
    arma::vec s, y;
    u_long k = 0;
    do {
        if (k >= _max_iter) {
            _exit_flag = 1;
            _n_iter += k;
            return;
        }
        p = -(_H*_g);
        if (p.has_nan() || p.has_inf()) {
            _H = hessian(x);
            bool chol_success = arma::inv_sympd(_H,_H);
            if (!chol_success) _H = arma::pinv(_H);
            p = -(_H*_g);
        }
        alpha = wolfe_step(f,grad_f,x,p,_wolfe_c1,_wolfe_c2);
        s = alpha*p;

        if (s.has_nan() || s.has_inf()) {
            _n_iter += k;
            _exit_flag = 2;
            return;
        }

        x += s;
        g1 = grad_f(x);
        y = g1 - _g;

        double sdoty = arma::dot(s,y);
        arma::vec Hdoty = _H*y;
        _H += (1 + arma::dot(y,Hdoty) / sdoty) * (s*s.t())/sdoty - (s*Hdoty.t() + Hdoty*s.t())/sdoty;
        _g = g1;
        k++;
    } while (arma::norm(p,"inf") > _tol);
    _n_iter += k;
    _exit_flag = 0;
}