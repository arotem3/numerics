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

numerics::Polynomial::Polynomial(Polynomial&& P) : a(_a), b(_b), degree(_deg), coefficients(_p) {
    _p = std::move(P._p);
    _deg = std::move(P._deg);
    _a = std::move(P._a);
    _b = std::move(P._b);
}

numerics::Polynomial::Polynomial(double s) : a(_a), b(_b), coefficients(_p), degree(_deg) {
    _p = {s};
    _a = -1;
    _b = 1;
    _set_degree();
}

numerics::Polynomial::Polynomial(const Polynomial& P) : a(_a), b(_b), coefficients(_p), degree(_deg) {
    _p = P._p;
    _a = P._a;
    _b = P._b;
    _set_degree();
}

void numerics::Polynomial::operator=(const Polynomial& P) {
    _p = P._p;
    _a = P._a;
    _b = P._b;
    _set_degree();
}

numerics::Polynomial::Polynomial(const arma::mat& p, double aa, double bb) : a(_a), b(_b), coefficients(_p), degree(_deg) {
    _p = p;
    _a = aa;
    _b = bb;
    _set_degree();
}

numerics::Polynomial::Polynomial(arma::mat&& p, double aa, double bb) : a(_a), b(_b), coefficients(_p), degree(_deg) {
    _p = p;
    _a = aa;
    _b = bb;
    _set_degree();
}

numerics::Polynomial::Polynomial(const arma::vec& x, const arma::mat& y, u_int deg) : a(_a), b(_b), coefficients(_p), degree(_deg) {
    _p.set_size(deg+1, y.n_cols);
    _a = x.min();
    _b = x.max();
    arma::vec xx = 2*(x - _a)/(_b - _a) - 1;
    for (u_long i=0; i < y.n_cols; ++i) _p.col(i) = arma::polyfit(xx, y.col(i), deg);
    _set_degree();
}

numerics::Polynomial::Polynomial(const arma::vec& x, const arma::mat& y) : a(_a), b(_b), coefficients(_p), degree(_deg) {
    u_int n = x.n_elem;
    _p.set_size(n, y.n_cols);
    _a = x.min();
    _b = x.max();
    arma::vec xx = 2*(x - _a)/(_b - _a) - 1;
    for (u_long i=0; i < y.n_cols; ++i) _p.col(i) = arma::polyfit(xx, y.col(i), n-1);
    _set_degree();
}

arma::mat numerics::Polynomial::operator()(const arma::vec& x) const {
    arma::mat out(x.n_elem, _p.n_cols);
    arma::vec xx = 2*(x - _a)/(_b - _a) - 1;
    for (u_long i=0; i < _p.n_cols; ++i) out.col(i) = arma::polyval(_p.col(i), xx);
    return out;
}

numerics::Polynomial numerics::Polynomial::transform(double aa, double bb) const {
    double c = (bb - aa) / (_b - _a);
    double d = aa - c*_a;
    
    arma::mat T = arma::zeros(degree + 1, degree + 1);
    for (u_long i=0; i < degree+1; ++i) {
        for (u_long j=i; j < degree+1; ++j) {
            double jchoosei = (i < j-i) ? (i * std::beta(i, j-i+1)) : ((j-i)*std::beta(i+1, j-i));
            jchoosei = 1 / jchoosei;
            T(i, j) = jchoosei * std::pow(c, i) * std::pow(d, j-i);
        }
    }

    arma::mat pp = arma::reverse(T * arma::reverse(_p));
    return Polynomial(std::move(pp), aa, bb);
}

numerics::Polynomial numerics::Polynomial::derivative(u_int k) const {
    arma::mat dp(_p.n_rows-k, _p.n_cols);
    for (u_long i=0; i < _p.n_cols; ++i) dp.col(i) = polyder(_p, k) * 2 / (_b - _a);
    return Polynomial(std::move(dp), _a, _b);
}

numerics::Polynomial numerics::Polynomial::integral() const {
    arma::mat ip(_p.n_rows+1, _p.n_cols);
    for (u_long i=0; i < _p.n_cols; ++i) ip.col(i) = polyint(_p.col(i))*(_b - _a) / 2;
    return Polynomial(std::move(ip), _a, _b);
}

numerics::Polynomial numerics::Polynomial::operator+(const Polynomial& P) const {
    if (P.coefficients.n_cols != _p.n_cols) {
        throw std::runtime_error("Polynomial addition error: vector polynomials must have the same number of components");
    }
    Polynomial PP = P.transform(_a, _b);
    u_long k = std::max(PP.degree, degree) + 1;
    arma::mat pplus = arma::zeros(k, _p.n_cols);
    pplus.tail_rows(degree+1) += _p;
    pplus.tail_rows(PP.degree+1) += PP._p;
    return Polynomial(std::move(pplus), _a, _b);
}

numerics::Polynomial numerics::Polynomial::operator+(double c) const {
    Polynomial pplus(_p, _a, _b);
    pplus._p.tail_rows(1) += c;
    return pplus;
}

numerics::Polynomial numerics::Polynomial::operator-() const {
    return Polynomial(-_p, _a, _b);
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
    if (P.coefficients.n_cols != _p.n_cols) {
        throw std::runtime_error("Polynomial multiplication error: vector polynomials must have the same number of components");
    }
    Polynomial PP = P.transform(_a, _b);
    arma::mat pprod(_p.n_rows + PP._p.n_rows, _p.n_cols);
    for (u_long i=0; i < _p.n_cols; ++i) pprod.col(i) = arma::conv(_p, PP._p, "full");
    return Polynomial(std::move(pprod), _a, _b);
}

numerics::Polynomial numerics::Polynomial::operator*(double c) const {
    return Polynomial(_p*c, _a, _b);
}

numerics::Polynomial& numerics::Polynomial::operator+=(const Polynomial& P) {
    Polynomial pplus = (*this) + P.transform(_a, _b);
    _p = std::move(pplus._p);
    _set_degree();
    return *this;
}

numerics::Polynomial& numerics::Polynomial::operator+=(double c) {
    _p.tail_rows(1) += c;
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
    _p = arma::conv(_p, P.transform(_a, _b)._p, "full");
    _set_degree();
    return *this;
}

numerics::Polynomial& numerics::Polynomial::operator*=(double c) {
    _p *= c;
    return *this;
}

std::ostream& numerics::operator<<(std::ostream& out, const numerics::Polynomial& p) {
    for (int i=0; i < p.degree+1; ++i) {
        out << p.coefficients.row(i);
        if (i < p.degree) out << " * (" << 2/(p.b-p.a) << " * x + " << 1 - p.a / (p.b - p.a) << ")";
        if (i < p.degree-1) out << "^" << p.degree-i;
        if (i < p.degree) out << " + ";
    }
    return out;
}

numerics::PieceWisePoly numerics::PieceWisePoly::derivative(int k) const {
    PieceWisePoly out(_extrapolation_type(_extrap), _extrap_val, _dim);
    for (const auto& el : _pieces) {
        out.push(el.second.derivative());
    }
    return out;
}

numerics::PieceWisePoly numerics::PieceWisePoly::integral() const {
    PieceWisePoly out(_extrapolation_type(_extrap), _extrap_val, _dim);
    arma::rowvec C = arma::zeros(out._dim);
    for (const auto& el : _pieces) {
        Polynomial poly = el.second.integral();
        C -= poly(arma::vec({poly.a}));
        poly += C;
        C = poly(arma::vec({poly.b}));
        out.push(std::move(poly));
    }
    return out;
}