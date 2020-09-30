#ifndef NUMERICS_INTERPOLATION_HPP
#define NUMERICS_INTERPOLATION_HPP

arma::vec polyder(const arma::vec& p, uint k = 1);
arma::vec polyint(const arma::vec& p, double c = 0);

class Polynomial {
    protected:
    arma::vec _p;

    public:
    const arma::vec& coefficients;

    // default constructor --> constant
    explicit Polynomial(double s=0) : coefficients(_p) {
        _p = {s};
    }

    Polynomial(const Polynomial& P) : coefficients(_p) {
        _p = P._p;
    }

    void operator=(const Polynomial& P) {
        _p = P._p;
    }

    // initialize with polynomial coefficients taken column-wise
    explicit Polynomial(const arma::vec& p) : coefficients(_p) {
        _p = p;
    }

    explicit Polynomial(arma::vec&& p) : coefficients(_p) {
        _p = p;
    }
    
    // initialize with interpolation problem
    explicit Polynomial(const arma::vec& x, const arma::vec& y) : coefficients(_p) {
        u_int n = x.n_elem;
        _p = arma::polyfit(x, y, n-1);
    }

    double operator()(double x) const {
        arma::vec t = {x};
        t = polyval(_p, t);
        return t(0);
    }

    arma::vec operator()(const arma::vec& x) const {
        return arma::polyval(_p, x);
    }
    Polynomial derivative(u_int k=1) const {
        Polynomial P(polyder(_p,k));
        return P;
    }
    Polynomial integral(double c=0) const {
        Polynomial P(polyint(_p, c));
        return P;
    }

    Polynomial operator+(const Polynomial& P) const {
        if (_p.n_elem < P._p.n_elem) {
            Polynomial pplus(P._p);
            pplus._p.head(_p.n_rows) += _p;
            return pplus;
        } else {
            Polynomial pplus(_p);
            pplus._p.head(P._p.n_rows) += P._p;
            return pplus;
        }
    }
    Polynomial operator+(double c) const {
        Polynomial P(_p);
        P._p(0) += c;
        return P;
    };

    Polynomial operator-() const {
        Polynomial P(-_p);
        return P;
    }
    Polynomial operator-(const Polynomial& P) const {
        Polynomial Pm(_p);
        Pm._p -= P._p;
        return Pm;
    }
    Polynomial operator-(double c) const {
        Polynomial P(_p);
        P._p(0) -= c;
        return P;
    }

    Polynomial operator*(const Polynomial& P) const {
        Polynomial PtimesP(arma::conv(_p, P._p, "full"));
        return PtimesP;
    }
    Polynomial operator*(double c) const {
        Polynomial PtimesC(_p*c);
        return PtimesC;
    }
};

class PieceWisePoly {
    protected:
    double _lb, _ub;
    short _extrap;
    double _extrap_val;
    std::vector<Polynomial> _P;
    arma::vec _x;

    void _check_xy(const arma::vec& x, const arma::vec& y) {
        if (x.n_elem != y.n_elem) {
            throw std::invalid_argument("dimension mismatch, x.n_elem (=" + std::to_string(x.n_elem) + ") != y.n_rows (=" + std::to_string(y.n_elem) + ")");
        }
    }

    void _check_x(const arma::vec& x) { // verify no reps in sorted array
        for (u_long i=0; i < x.n_elem-1; ++i) {
            if (x(i) == x(i+1)) {
                throw std::runtime_error("one or more x values are repeting, therefore no cubic interpolation exists for this data");
            }
        }
        _lb = x.front(); _lb -= 1e-8*std::abs(_lb);
        _ub = x.back(); _ub += 1e-8*std::abs(_ub);
    }

    double _periodic(double t) const {
        if (t == _ub) return _ub;
        else {
            double q = (t - _lb)/(_ub - _lb);
            q = q - std::floor(q);
            q = (_ub - _lb) * q + _lb;
            return q;
        }
    }

    double _flat_past_boundary(double t) const {
        if (t <= _lb) return _lb;
        else if (t >= _ub) return _ub;
        else return t;
    }

    public:
    PieceWisePoly(const std::string& extrapolation="const", double val=0) {
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

    double operator()(double t) const {
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
    
    arma::vec operator()(const arma::vec& t) const {
        arma::vec yh = arma::vec(t.n_elem);
        for (u_long i=0; i < t.n_elem; ++i) {
            yh(i) = (*this)(t(i));
        }
        return yh;
    }

    PieceWisePoly derivative(int k=1) const {
        PieceWisePoly out = *this;
        for (u_long i=0; i < _P.size(); ++i) {
            out._P.at(i) = out._P.at(i).derivative();
        }
        return out;
    }

    PieceWisePoly integral(double c=0) const {
        PieceWisePoly out = *this;
        for (u_long i=0; i < _P.size(); ++i) {
            out._P.at(i) = out._P.at(i).integral();
        }
        return out;
    }
};

class CubicInterp : public PieceWisePoly {
    public:
    CubicInterp(const arma::vec& x, const arma::vec& y, const std::string& extrapolation="boundary", double val=0);
};

class HSplineInterp : public PieceWisePoly {
    public:
    HSplineInterp(const arma::vec& x, const arma::vec& y, const std::string& extrapolation="linear", double val=0);
    HSplineInterp(const arma::vec& x, const arma::vec& y, const arma::vec& yp, const std::string& extrapolation="linear", double val=0);
};

arma::mat lagrange_interp(const arma::vec&, const arma::mat&, const arma::vec&);
arma::mat sinc_interp(const arma::vec&, const arma::mat&, const arma::vec&);

#endif