#ifndef NUMERICS_INTERPOLATION_HPP
#define NUMERICS_INTERPOLATION_HPP

/* polyder(p, k) : return the k^th derivative of a polynomial.
 * --- p : polynomial to differentiate.
 * --- k : the derivative order (k = 1 by default, i.e. first derivative). */
arma::vec polyder(const arma::vec& p, uint k = 1);

/* polyint(p, c) : return the integral of a polynomial.
 * --- p : polynomial to integrate.
 * --- c : integration constant. */
arma::vec polyint(const arma::vec& p, double c = 0);

class Polynomial {
    protected:
    arma::vec _p;
    u_int _deg;

    void _set_degree();

    public:
    const u_int& degree;
    const arma::vec& coefficients;

    // default constructor --> constant
    explicit Polynomial(double s=0);
    // initialize with polynomial coefficients taken column-wise
    explicit Polynomial(const arma::vec& p);
    explicit Polynomial(arma::vec&& p);
    // initialize with interpolation problem
    explicit Polynomial(const arma::vec& x, const arma::vec& y, u_int deg);
    explicit Polynomial(const arma::vec& x, const arma::vec& y);

    Polynomial(const Polynomial& P);
    void operator=(const Polynomial& P);

    double operator()(double x) const;
    arma::vec operator()(const arma::vec& x) const;


    Polynomial derivative(u_int k=1) const;
    Polynomial integral(double c=0) const;
    Polynomial operator+(const Polynomial& P) const;
    Polynomial operator+(double c) const;
    
    Polynomial operator-() const;
    Polynomial operator-(const Polynomial& P) const;
    Polynomial operator-(double c) const;
    

    Polynomial operator*(const Polynomial& P) const;
    Polynomial operator*(double c) const;

    Polynomial& operator+=(const Polynomial& P);
    Polynomial& operator+=(double c);
    Polynomial& operator-=(const Polynomial& P);
    Polynomial& operator-=(double c);
    Polynomial& operator*=(const Polynomial& P);
    Polynomial& operator*=(double c);
};

std::ostream& operator<<(std::ostream& out, const Polynomial& p);

class PieceWisePoly {
    friend PieceWisePoly natural_cubic_spline(const arma::vec& x, const arma::vec&y, const std::string& extrapolation, double val);
    friend PieceWisePoly hermite_cubic_spline(const arma::vec& x, const arma::vec& y, const std::string& extrapolation, double val);
    friend PieceWisePoly hermite_cubic_spline(const arma::vec& x, const arma::vec& y, const arma::vec& yp, const std::string& extrapolation, double val);
    
    protected:
    double _lb, _ub;
    short _extrap;
    double _extrap_val;
    std::vector<Polynomial> _P;
    arma::vec _x;

    void _check_xy(const arma::vec& x, const arma::vec& y);
    void _check_x(const arma::vec& x);

    double _periodic(double t) const;
    double _flat_past_boundary(double t) const;

    public:
    PieceWisePoly(const std::string& extrapolation="const", double val=0);

    double operator()(double t) const;
    
    arma::vec operator()(const arma::vec& t) const;

    PieceWisePoly derivative(int k=1) const;
    PieceWisePoly integral(double c=0) const;
};

PieceWisePoly natural_cubic_spline(const arma::vec& x, const arma::vec&y, const std::string& extrapolation="boundary", double val=0);
PieceWisePoly hermite_cubic_spline(const arma::vec& x, const arma::vec& y, const std::string& extrapolation="linear", double val=0);
PieceWisePoly hermite_cubic_spline(const arma::vec& x, const arma::vec& y, const arma::vec& yp, const std::string& extrapolation="linear", double val=0);

arma::mat lagrange_interp(const arma::vec&, const arma::mat&, const arma::vec&);
arma::mat sinc_interp(const arma::vec&, const arma::mat&, const arma::vec&);

#endif