#include <numerics.hpp>

void numerics::optimization::TrustNewton::_initialize(const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) {
    _F = f(x);
    _delta = 0.1*std::max(1.0, arma::norm(x,"inf"));
}

bool numerics::optimization::TrustNewton::_step(arma::vec& dx, arma::vec& F1, const arma::vec& x, const VecFunc& f, const MatFunc* jacobian) {
    arma::vec g, p, Jg;
    bool success;

    double delta_max = 10.0;

    if (jacobian == nullptr) {
        auto fnorm = [&f](const arma::vec& z) -> double {arma::vec fz = f(z); return 0.5*arma::dot(fz,fz);};
        g = grad(fnorm, x);

        double h = 1e-6*std::max(1.0, arma::norm(x));
        VecFunc JacMult = [this, h, &F1, &f, &x](const arma::vec& v) -> arma::vec {
            double C = std::max(1.0, arma::norm(v)) / h;
            return C * (f(x + v/C) - _F);
        };
        success = gmres(p, JacMult, _F, _xtol);
        Jg = JacMult(g);
    } else {
        _J = (*jacobian)(x);
        g = _J.t() * _F;
        success = arma::solve(p, _J, _F);
        Jg = _J * g;
    }

    if (not success) return false;
    
    double ff = arma::dot(_F,_F);
    double gg = arma::dot(g,g);
    double Jg2 = arma::dot(Jg,Jg);
    double pp = arma::dot(p,p);

    arma::mat A = {
        {ff,gg},
        {gg,Jg2}
    };
    arma::mat B = {
        {pp,ff},
        {ff,gg}
    };

    arma::vec r = {ff, gg};

    while (true) {
        if (arma::norm(p) < _delta) {
            dx = -p;
            _fh = -0.5*arma::dot(g,p);
        }
        else {
            arma::vec u;
            auto phi = [this,&u,&A,&B,&r,&p,&g](double l) -> double {
                u = arma::solve(A + (l*l)*B, -r);
                double unorm = arma::norm(u(0)*p + u(1)*g);
                return 1/unorm - 1/_delta;
            };
            double l = newton_1d(phi, 1.0, 1e-4);
            l *= l;
            dx = u(0)*p + u(1)*g;
            _fh = arma::dot(u, 0.5*A*u + r);
        }

        F1 = f(x + dx);
        double f1 = arma::dot(F1,F1);

        if (f1 < ff) {
            double rho = 0.5*std::abs((ff - f1) / _fh);
            if (rho < 0.25) {
                _delta = 0.25*arma::norm(dx);
            } else if ((rho > 0.75) and (std::abs(arma::norm(dx)-_delta) < 1e-8)) {
                // approximation is good and dx is on the boundary of the trust region
                _delta = std::min(2*_delta, delta_max*arma::norm(x));
            }
            break;
        } else {
            _delta /= 2.0;
        }
        if (_delta < _xtol) break;
    }
    return success;
}

void numerics::optimization::TrustMin::_initialize(const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) {
    _g = df(x);
    _f0 = f(x);
    _delta = 0.1*std::max(1.0, arma::norm(x,"inf"));
}

bool numerics::optimization::TrustMin::_step(arma::vec& dx, arma::vec& g1, const arma::vec& x, const dFunc& f, const VecFunc& df, const MatFunc* hessian) {
    bool success;
    double delta_max = 10.0;

    arma::vec p, Hg;

    if (hessian == nullptr) {
        double h = 1e-6*std::max(1.0, arma::norm(x));
        VecFunc HessMult = [this,&df,&x,h](const arma::vec& v) -> arma::vec {
            double C = std::max(1.0, arma::norm(v)) / h;
            return C * (df(x + v/C) - _g);
        };
        Hg = HessMult(_g);
        success = pcg(p, HessMult, -_g, _xtol);
        if (not success) success = gmres(p, HessMult, _g, _xtol);
    } else {
        arma::mat H = (*hessian)(x);
        Hg = H*_g;
        success = arma::solve(p, H, _g);
    }

    if (not success) return false;

    double gp = arma::dot(_g,p);
    double gg = arma::dot(_g,_g);
    double pp = arma::dot(p,p);
    double gHg = arma::dot(_g,Hg);

    arma::mat A = {
        {gp, gg},
        {gg, gHg}
    };

    arma::mat B = {
        {pp, gp},
        {gp, gg}
    };

    arma::vec r = {gp, gg};

    while (true) {
        if (arma::norm(p) < _delta) {
            dx = -p;
            _fh = -0.5*arma::dot(_g,p);
        }
        else {
            arma::vec u;
            auto phi = [this,&u,&A,&B,&r,&p](double l) -> double {
                l = l*l;
                u = arma::solve(A + l*B, -r);
                double unorm = arma::norm(u(0)*p + u(1)*_g);
                return 1/unorm - 1/_delta;
            };
            double l = newton_1d(phi, 1.0, 1e-4);
            l = l*l;
            dx = u(0)*p + u(1)*_g;
            _fh = arma::dot(u, 0.5*A*u + r);
        }

        double f1 = f(x+dx);

        if (f1 < _f0) {
            double rho = std::abs((_f0 - f1) / _fh);
            if (rho < 0.25) {
                _delta = 0.25*arma::norm(dx);
            } else if ((rho > 0.75) and (std::abs(arma::norm(dx)-_delta) < 1e-8)) {
                // approximation is good and dx is on the boundary of the trust region
                _delta = std::min(2*_delta, delta_max*arma::norm(x));
            }
            _f0 = f1;
            g1 = df(x+dx);
            break;
        } else {
            _delta /= 2.0;
        }
        
        if (_delta < _xtol) {
            dx = arma::zeros(arma::size(x));
            break;
        }
    }
    return true;
}