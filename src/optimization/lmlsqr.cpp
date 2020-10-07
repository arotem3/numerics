#include <numerics.hpp>

void numerics::optimization::LmLSQR::fsolve(arma::vec& x, const VecFunc& f) {
    _check_loop_parameters();
    double tau = _damping_param, nu = _damping_scale;
    arma::vec F1, delta = 0.01*arma::ones(arma::size(x));

    _J = approx_jacobian(f,x);
    _F = f(x);

    arma::vec D; arma::mat V;
    if (!_use_cgd) arma::eig_sym(D, V, _J.t() * _J);
    double lam = tau*arma::norm(_J.diag(),"inf");

    u_long k = 0;
    VerboseTracker T(_max_iter);
    if (_v) T.header("|f|_2");
    do {
        if (k >= _max_iter) {
            _exit_flag = 1;
            _n_iter += k;
            if (_v) T.max_iter_flag();
            return;
        }

        if (_v) T.iter(k, arma::norm(_F));

        arma::vec RHS = -(_J.t() * _F);
        double rho;
        do {
            arma::mat LHS;
            if (_use_cgd) {
                LHS = _J.t() * _J;
                LHS.diag() += lam;
                cgd(delta, LHS, RHS);
            } else {
                delta = V * arma::diagmat(1/(D+lam)) * V.t() * RHS;
            }
            
            if (delta.has_nan() || delta.has_inf()) {
                _exit_flag = 2;
                _n_iter += k;
                if (_v) T.nan_flag();
                return;
            }
            F1 = f(x + delta);

            rho = (arma::norm(_F) - arma::norm(F1));
            rho /= arma::dot(delta, lam*delta + RHS);
            if (rho > 0) { // new point is acceptible
                x += delta;
                _J += ((F1-_F) - _J*delta)*delta.t() / arma::dot(delta, delta);
                if (_J.has_nan() || _J.has_inf()) _J = approx_jacobian(f,x);
                if (!_use_cgd) arma::eig_sym(D, V, _J.t() * _J);
                _F = F1;
                lam *= std::max( 0.33, 1 - std::pow(2*rho-1,3) ); // 1 - (2r-1)^3
                nu = 2;
            } else { // need to shrink trust region
                lam *= nu;
                nu *= 2;
            }
            k++;
        } while(rho < 0);
    } while (arma::norm(delta,"inf") > _tol);

    _n_iter += k;
    _exit_flag = 0;
    if (_v) T.success_flag();
}

void numerics::optimization::LmLSQR::fsolve(arma::vec& x, const VecFunc& f, const MatFunc& jacobian) {
    _check_loop_parameters();
    double tau = _damping_param, nu = _damping_scale;
    arma::vec F1, delta = 0.01*arma::ones(arma::size(x));

    _J = jacobian(x);
    _F = f(x);

    double lam = tau*arma::norm(_J.diag(),"inf");
    arma::mat V; arma::vec D;
    if (!_use_cgd) arma::eig_sym(D, V, _J.t()*_J);

    u_long k = 0;
    VerboseTracker T(_max_iter);
    if (_v) T.header("|f|_2");
    do {
        if (k >= _max_iter) {
            _exit_flag = 1;
            _n_iter += k;
            if (_v) T.max_iter_flag();
            return;
        }

        if (_v) T.iter(k, arma::norm(_F));

        arma::vec RHS = -(_J.t() * _F);
        double rho;
        do {
            arma::mat LHS;
            if (_use_cgd) {
                LHS = _J.t() * _J;
                LHS.diag() += lam;
                cgd(delta, LHS, RHS);
            } else {
                delta = V * arma::diagmat(1/(D+lam)) * V.t() * RHS;
            }
            
            if (delta.has_nan() || delta.has_inf()) {
                _exit_flag = 2;
                _n_iter += k;
                if (_v) T.nan_flag();
                return;
            }
            F1 = f(x + delta);

            rho = (arma::norm(_F) - arma::norm(F1));
            rho /= arma::dot(delta, lam*delta + RHS);
            if (rho > 0) {
                x += delta;
                _J = jacobian(x);
                if (!_use_cgd) arma::eig_sym(D, V, _J.t()*_J);
                _F = F1;
                lam *= std::max( 0.33, 1 - std::pow(2*rho-1,3) ); // 1 - (2r-1)^3
                nu = 2;
            } else {
                lam *= nu;
                nu *= 2;
            }
            k++;
        } while(rho < 0);
    } while (arma::norm(delta,"inf") > _tol);

    _n_iter += k;
    _exit_flag = 0;
    if (_v) T.success_flag();
}