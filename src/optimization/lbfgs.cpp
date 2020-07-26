#include <numerics.hpp>

/* lbfgs_update(p,S,Y) : update function for limited memory BFGS
 * --- p : negative gradient, the search direction is also stored here
 * --- S : s_i = x_i - x_(i-1) 
 * --- Y : y_i = f_i - f_(i-1) */
void numerics::optimization::LBFGS::_lbfgs_update(arma::vec& p) {
    u_long k = _S.size();

    arma::vec ro = arma::zeros(k);
    for (u_long i=0; i < k; ++i) {
        ro(i) = 1 / arma::dot(_S.at(i),_Y.at(i));
    }

    arma::vec q = p;
    arma::vec alpha = arma::zeros(k);
    
    for (int i(k-1); i >= 0; --i) {
        alpha(i) = ro(i) * arma::dot(_S.at(i),q);
        q -= alpha(i) * _Y.at(i);
    }

    arma::vec r = q * _hdiag;

    for (int i(0); i < k; ++i) {
        double beta = ro(i) * arma::dot(_Y.at(i),r);
        r += _S.at(i) * (alpha(i) - beta);
    }

    p = r;
}

void numerics::optimization::LBFGS::minimize(arma::vec& x, const dFunc& f, const VecFunc& grad_f) {
    _check_loop_parameters();
    u_long n = x.n_elem;
    arma::vec x1, g1, s, y, p;
    double alpha;

    VerboseTracker T(_max_iter);
    if (_v) {
        T.header();
        T.iter(0, f(x));
    }

    _g = grad_f(x);
    p = -_g;
    alpha = wolfe_step(f, grad_f, x, p, _wolfe_c1, _wolfe_c2);
    x1 = x + alpha * p;
    _S.clear();
    _Y.clear();

    u_long iters = 1;
    while (true) {
        if (arma::norm(_g, "inf") < _tol) {
            _exit_flag = 0;
            _n_iter += iters;
            if (_v) T.success_flag();
            return;
        }

        if (iters >= _max_iter) {
            _exit_flag = 1;
            _n_iter += iters;
            if (_v) T.max_iter_flag();
            return;
        }

        if (_v) T.iter(iters, f(x));

        g1 = grad_f(x1);

        s = x1 - x;
        y = g1 - _g;
        _hdiag = arma::dot(s, y) / arma::dot(y, y);

        _S.push(s);
        _Y.push(y);

        p = -g1;
        
        _lbfgs_update(p);
        if (p.has_nan() || p.has_inf()) {
            p = -grad_f(x);
            _S.clear();
            _Y.clear();
        }

        alpha = wolfe_step(f, grad_f, x1, p, _wolfe_c1, _wolfe_c2);

        if (std::isnan(alpha) || std::isinf(alpha)) {
            _exit_flag = 2;
            _n_iter += iters;
            if (_v) T.nan_flag();
            return;
        }

        x = x1; _g = g1;
        x1 += alpha * p;

        iters++;
    }
}