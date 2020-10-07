#include "numerics.hpp"

arma::vec numerics::polyder(const arma::vec& p, uint k) {
    if (p.n_elem <= 1) return arma::zeros(1);
    int n = p.n_elem - 1;
    arma::vec dp = arma::zeros(n);
    for (uint i=0; i < n; ++i) {
        dp(i) = (n-i) * p(i);
    }
    if (k > 1) return polyder(dp, k-1);
    else return dp;
}

arma::vec numerics::polyint(const arma::vec& p, double c) {
    int n = p.n_elem + 1;
    arma::vec ip = arma::zeros(n);
    for (uint i=0; i < n-1; ++i) {
        ip(i) = p(i) / (n-1-i);
    }
    ip(n-1) = c;
    return ip;
}

void numerics::Polynomial::_set_degree() {
    _deg = _p.n_elem - 1;
}

numerics::Polynomial::Polynomial(double s) : coefficients(_p), degree(_deg) {
    _p = {s};
    _set_degree();
}

numerics::Polynomial::Polynomial(const Polynomial& P) : coefficients(_p), degree(_deg) {
    _p = P._p;
    _set_degree();
}

void numerics::Polynomial::operator=(const Polynomial& P) {
    _p = P._p;
    _set_degree();
}

numerics::Polynomial::Polynomial(const arma::vec& p) : coefficients(_p), degree(_deg) {
    _p = p;
    _set_degree();
}

numerics::Polynomial::Polynomial(arma::vec&& p) : coefficients(_p), degree(_deg) {
    _p = p;
    _set_degree();
}

numerics::Polynomial::Polynomial(const arma::vec& x, const arma::vec& y, u_int deg) : coefficients(_p), degree(_deg) {
    _p = arma::polyfit(x, y, deg);
    _set_degree();
}

numerics::Polynomial::Polynomial(const arma::vec& x, const arma::vec& y) : coefficients(_p), degree(_deg) {
    u_int n = x.n_elem;
    _p = arma::polyfit(x, y, n-1);
    _set_degree();
}

double numerics::Polynomial::operator()(double x) const {
    arma::vec t = {x};
    t = polyval(_p, t);
    return t(0);
}

arma::vec numerics::Polynomial::operator()(const arma::vec& x) const {
    return arma::polyval(_p, x);
}

numerics::Polynomial numerics::Polynomial::derivative(u_int k) const {
    // Polynomial P(polyder(_p,k));
    return Polynomial(polyder(_p,k));
}

numerics::Polynomial numerics::Polynomial::integral(double c) const {
    // Polynomial P(polyint(_p, c));
    return Polynomial(polyint(_p,c));
}

numerics::Polynomial numerics::Polynomial::operator+(const Polynomial& P) const {
    u_long k = std::max(P.degree, degree) + 1;
    arma::vec pplus = arma::zeros(k);
    pplus.tail(degree+1) += _p;
    pplus.tail(P.degree+1) += P._p;
    return Polynomial(std::move(pplus));
}

numerics::Polynomial numerics::Polynomial::operator+(double c) const {
    Polynomial pplus(_p);
    pplus._p.tail(1) += c;
    return pplus;
}

numerics::Polynomial numerics::Polynomial::operator-() const {
    return Polynomial(-_p);
}

numerics::Polynomial numerics::Polynomial::operator-(const Polynomial& P) const {
    Polynomial pminus = (*this) + (-P);
    return pminus;
}

numerics::Polynomial numerics::Polynomial::operator-(double c) const {
    Polynomial pminus = (*this) + (-c);
    return pminus;
}

numerics::Polynomial numerics::Polynomial::operator*(const Polynomial& P) const {
    return Polynomial(arma::conv(_p, P._p, "full"));
}

numerics::Polynomial numerics::Polynomial::operator*(double c) const {
    return Polynomial(_p*c);
}

numerics::Polynomial& numerics::Polynomial::operator+=(const Polynomial& P) {
    Polynomial pplus = (*this) + P;
    _p = std::move(pplus._p);
    _set_degree();
    return *this;
}

numerics::Polynomial& numerics::Polynomial::operator+=(double c) {
    _p.tail(1) += c;
    return *this;
}

numerics::Polynomial& numerics::Polynomial::operator-=(const Polynomial& P) {
    (*this) += (-P);
    return *this;
}

numerics::Polynomial& numerics::Polynomial::operator-=(double c) {
    (*this) += (-c);
    return *this;
}

numerics::Polynomial& numerics::Polynomial::operator*=(const Polynomial& P) {
    _p = arma::conv(_p,P._p, "full");
    _set_degree();
    return *this;
}

numerics::Polynomial& numerics::Polynomial::operator*=(double c) {
    _p *= c;
    return *this;
}

std::ostream& numerics::operator<<(std::ostream& out, const numerics::Polynomial& p) {
    for (int i=0; i < p.degree+1; ++i) {
        out << p.coefficients(i);
        if (i < p.degree) out << " * x";
        if (i < p.degree-1) out << "^" << p.degree-i;
        if (i < p.degree) out << " + ";
    }
    return out;
}

void numerics::PieceWisePoly::_check_xy(const arma::vec& x, const arma::vec& y) {
    if (x.n_elem != y.n_elem) {
        throw std::invalid_argument("dimension mismatch, x.n_elem (=" + std::to_string(x.n_elem) + ") != y.n_rows (=" + std::to_string(y.n_elem) + ")");
    }
}

void numerics::PieceWisePoly::_check_x(const arma::vec& x) { // verify no reps in sorted array
    for (u_long i=0; i < x.n_elem-1; ++i) {
        if (x(i) == x(i+1)) {
            throw std::runtime_error("one or more x values are repeting, therefore no cubic interpolation exists for this data");
        }
    }
    _lb = x.front(); _lb -= 1e-8*std::abs(_lb);
    _ub = x.back(); _ub += 1e-8*std::abs(_ub);
}

double numerics::PieceWisePoly::_periodic(double t) const {
    if (t == _ub) return _ub;
    else {
        double q = (t - _lb)/(_ub - _lb);
        q = q - std::floor(q);
        q = (_ub - _lb) * q + _lb;
        return q;
    }
}

double numerics::PieceWisePoly::_flat_past_boundary(double t) const {
    if (t <= _lb) return _lb;
    else if (t >= _ub) return _ub;
    else return t;
}

numerics::PieceWisePoly::PieceWisePoly(const std::string& extrapolation, double val) {
    if (extrapolation == "const") {
        _extrap = 0;
        _extrap_val = val;
    }
    else if (extrapolation == "boundary") _extrap = 1;
    else if (extrapolation == "linear") _extrap = 2;
    else if (extrapolation == "periodic") _extrap = 3;
    else if (extrapolation == "polynomial") _extrap = 4;
    else throw std::invalid_argument("extrapolation type (=\"" + extrapolation + "\") must be one of {\"const\",\"linear\",\"periodic\",\"polynomial\"}.");
}

double numerics::PieceWisePoly::operator()(double t) const {
    if (_extrap == 3) t = _periodic(t); // periodic evaluation
    
    double out=0;
    if ((_lb <= t) and (t <= _ub)) {
        for (u_int i=1; i < _x.n_elem; ++i) {
            if ((_x(i-1) <= t) and (t <= _x(i))) out = _P.at(i-1)(t);
        }
    } else if (_extrap == 0) out = _extrap_val; // constant extrapolation set to value
    else if (_extrap == 1) {
        if (t < _lb) out = _P.front()(_lb);
        else out = _P.back()(_ub);
    } else if (_extrap == 2) { // linear extrapolation
        if (t < _lb) {
            double p0 = _P.front()(_lb);
            Polynomial dp = _P.front().derivative();
            double p1 = dp(_lb);
            out = p0 + p1*(t - _lb);
        } else {
            double p0 = _P.back()(_ub);
            Polynomial dp = _P.back().derivative();
            double p1 = dp(_ub);
            out = p0 + p1*(t - _ub);
        }
    } else { // polynomial extrapolation
        if (t < _lb) out = _P.front()(t);
        else out = _P.back()(t);
    }
    return out;
}

arma::vec numerics::PieceWisePoly::operator()(const arma::vec& t) const {
    arma::vec yh = arma::vec(t.n_elem);
    for (u_long i=0; i < t.n_elem; ++i) {
        yh(i) = (*this)(t(i));
    }
    return yh;
}

numerics::PieceWisePoly numerics::PieceWisePoly::derivative(int k) const {
    PieceWisePoly out = *this;
    for (u_long i=0; i < _P.size(); ++i) {
        out._P.at(i) = out._P.at(i).derivative();
    }
    return out;
}

std::string _extrapolation_type(short _extrap) {
    if (_extrap == 0) return "const";
    else if (_extrap == 1) return "boundary";
    else if (_extrap == 2) return "linear";
    else if (_extrap == 3) return "periodic";
    else return "polynomial";
}

numerics::PieceWisePoly numerics::PieceWisePoly::integral(double c) const {
    PieceWisePoly out(*this);
    double C = c;
    for (u_long i=0; i < out._P.size(); ++i) {
        Polynomial poly = out._P.at(i).integral();
        poly += (C - poly(_x.at(i))); // integrate, subtract value at lower bound of interval, add back C
        C = poly(_x.at(i+1)); // set C to value fo poly at upper bound as this will be the C for the next piece
        out._P.at(i) = std::move(poly);
    }
    return out;
}