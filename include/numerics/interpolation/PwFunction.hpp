#ifndef NUMERICS_INTERPOLATION_FWFUNCTION_HPP
#define NUMERICS_INTERPOLATION_FWFUNCTION_HPP

#include <map>
#include <string>

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

#ifdef NUMERICS_WITH_ARMA
namespace numerics
{

template<class T, typename Real=decltype(T::value_type)> // T must have T.a, T.b, arma::mat operator()(const arma::vec&), and copy constructor
class PwFunction
{
protected:
    enum _Extrapolation {
        CONST, // set to a single value
        BOUNDARY, // set to the value at boundaries (left or right)
        LINEAR, // extrapolate using a linear function at boundaries such that the extrapolation is C1
        PERIODIC, // periodically extend the function
        EXTEND // evaluates the boundary elements outside of their domain assuming the element function allows it.
    };

    std::map<Real, T> _pieces;
    Real _a, _b;
    u_long _dim;
    _Extrapolation _extrap;
    Real _extrap_val;

    // find p such that p.a < x < p.b
    inline const T& _find(Real x) const;
    inline void _set_ab();
    inline Real _periodic(Real t) const;

    inline std::string _extrapolation_type(numerics::_Extrapolation _extrap) const {
        if (_extrap == numerics::CONST) return "const";
        else if (_extrap == numerics::BOUNDARY) return "boundary";
        else if (_extrap == numerics::LINEAR) return "linear";
        else if (_extrap == numerics::PERIODIC) return "periodic";
        else return "extend";
    }


public:
    typedef Real value_type;

    const double& a = _a;
    const double& b = _b;

    inline PwFunction(PwFunction<T,Real>&& pwf);
    inline PwFunction(const PwFunction<T,Real>& pwf);
    inline explicit PwFunction(const std::string& extrapolation="const", double val=0, u_long dim=1);

    inline PwFunction<T,Real>& operator=(const PwFunction<T,Real>& f);
    inline PwFunction<T,Real>& operator=(PwFunction<T,Real>&& f);

    inline void push(const T& p);
    inline void push(T&& p);

    inline const T& find(Real x) const;

    arma::Mat<Real> operator()(const arma::Col<Real>& x) const;
};

template <class T, typename Real>
const T& PwFunction<T,Real>::_find(Real x) const {
    return (--_pieces.lower_bound(x))->second;
}

template <class T, typename Real>
void PwFunction<T,Real>::_set_ab() {
    _a = _pieces.begin()->second.a;
    _b = _pieces.rbegin()->second.b;
}

template <class T, typename Real>
Real PwFunction<T,Real>::_periodic(Real t) const {
    if (t == _b) return _b;
    else {
        double q = (t - _a)/(_b - _a);
        q = q - std::floor(q);
        q = (_b - _a) * q + _a;
        return q;
    }
}

template <class T, typename Real>
PwFunction<T,Real>::PwFunction(PwFunction<T,Real>&& pwf) {
    _pieces = std::move(pwf._pieces);
    _a = std::exchange(pwf._a, 0);
    _b = std::exchange(pwf._b, 0);
    _dim = std::exchange(pwf._dim, 0);
    _extrap = std::move(pwf._extrap);
    _extrap_val = std::exchange(pwf._extrap_val, 0);
}

template <class T, typename Real>
PwFunction<T,Real>::PwFunction(const std::string& extrapolation, double val, u_long dim) {
    if (extrapolation == "const") {
        _extrap = CONST;
        _extrap_val = val;
    }
    else if (extrapolation == "boundary") _extrap = BOUNDARY;
    else if (extrapolation == "linear") _extrap = LINEAR;
    else if (extrapolation == "periodic") _extrap = PERIODIC;
    else if (extrapolation == "extend") _extrap = EXTEND;
    else throw std::invalid_argument("extrapolation type (=\"" + extrapolation + "\") must be one of {\"const\",\"linear\",\"periodic\",\"polynomial\"}.");

    _dim = dim;
}

template <class T, typename Real>
PwFunction<T,Real>::PwFunction(const PwFunction<T,Real>& pwf) {
    _pieces = pwf._pieces;
    _a = pwf._a;
    _b = pwf._b;
    _dim = pwf._dim;
    _extrap = pwf._extrap;
    _extrap_val = pwf._extrap_val;
}

template <class T, typename Real>
PwFunction<T,Real>& PwFunction<T,Real>::operator=(const PwFunction<T,Real>& f) {
    _pieces = f._pieces;
    _a = f._a;
    _b = f._b;
    _dim = f._dim;
    _extrap = f._extrap;
    _extrap_val = f._extrap_val;

    return *this;
}

template <class T, typename Real>
PwFunction<T,Real>& PwFunction<T,Real>::operator=(PwFunction<T,Real>&& f) {
    _pieces = std::move(f._pieces);
    _a = std::move(f._a);
    _b = std::move(f._b);
    _dim = std::move(f._dim);
    _extrap = std::move(_extrap);
    _extrap_val = std::move(_extrap_val);

    return *this;
}

template <class T, typename Real>
void PwFunction<T,Real>::push(const T& p) {
    _pieces.emplace(p.a, p);
    _set_ab();
}

template <class T, typename Real>
void PwFunction<T,Real>::push(T&& p) {
    _pieces.emplace(p.a, p);
    _set_ab();
}

template <class T, typename Real>
const T& PwFunction<T,Real>::find(Real x) const {
    if ((_a < x) and (x < _b)) {
        return _find(x);
    }
    else {
        throw std::domain_error("PwFunction::find() error: domain is [" + std::to_string(_a) + ", " + std::to_string(_b) + "] but tried find piece containing x = " + std::to_string(x));
    }
}

template <class T, typename Real>
arma::Mat<Real> PwFunction<T,Real>::operator()(const arma::Col<Real>& x) const {
    arma::uvec i_valid = arma::find((_a < x) and (x < _b));

    arma::Mat<Real> y(x.n_elem, _dim);
    for (arma::uword& ii : i_valid) {
        y.row(ii) = _find(x(ii))(arma::Col<Real>({x(ii)}));
    }

    if ((_extrap == PERIODIC) or (_extrap == CONST)) {
        arma::uvec i_out = arma::find((x <= _a) or (_b <= x));
        
        if (_extrap == PERIODIC) {
            for (arma::uword& ii : i_out) {
                double t = _periodic(x(ii));
                y.row(ii) = _find(t)(arma::Col<Real>({t}));
            }
        }
        else if (_extrap == CONST) {
            y.rows(i_out).fill(_extrap_val);
        }
    }
    else {
        arma::uvec i_less = arma::find(x <= _a);
        arma::uvec i_greater = arma::find(x >= _b);
        
        if (_extrap == BOUNDARY) {
            arma::Row<Real> yL = _pieces.begin()->second(arma::Col<Real>({_a}));
            arma::Row<Real> yU = _pieces.rbegin()->second(arma::Col<Real>({_b}));
            y.each_row(i_less) = yL;
            y.each_row(i_greater) = yR;
        }
        else if (_extrap == EXTEND) {
            y.rows(i_less) = _pieces.begin()->second(x(i_less));
            y.rows(i_greater) = _pieces.rbegin()->second(x(i_greater));
        }
        else if (_extrap == LINEAR) {
            Real eps_a = std::max(1.0, std::abs(_a))*1e-6;
            arma::Mat<Real> fL = _pieces.begin()->second({_a, _a+eps_a});
            
            Real eps_b = std::max(1.0, std::abs(_b))*1e-6;
            arma::Mat<Real> fU = _pieces.rbegin()->second({_b-eps_b, _b});

            arma::Row<Real> dfL = (fL.row(1) - fL.row(0)) / eps_a;
            arma::Row<Real> dfU = (fU.row(1) - fU.row(0)) / eps_b; // approximate slope at boundary using centered difference

            arma::Row<Real> yL = fL.row(0);
            arma::Row<Real> yU = fU.row(1);

            y.rows(i_less) = arma::ones<arma::Col<Real>>(i_less.n_elem)*yL + (x(i_less) - _a)*dfL;
            y.rows(i_greater) = arma::ones<arma::Col<Real>>(i_greater.n_elem)*yU + (x(i_greater) - _b)*dfU;
        }
    }

    return y;
}

#ifdef NUMERICS_INTERPOLATION_POLYNOMIAL_HPP
template <typename Real>
class PieceWisePoly : public PwFunction<Polynomial<Real>> {
    public:
    PieceWisePoly(const std::string& extrapolation="const", double val=0, u_long dim=1) : PwFunction<Polynomial>(extrapolation, val, dim) {}

    #ifdef NUMERICS_INTERPOLATION_POLYDER_HPP
    PieceWisePoly derivative(int k=1) const {
        PieceWisePoly out(_extrapolation_type(_extrap), _extrap_val, _dim);
        for (const auto& el : _pieces) {
            out.push(el.second.derivative());
        }
        return out;
    }
    #endif

    #ifdef NUMERICS_INTERPOLATION_POLYINT_HPP
    PieceWisePoly integral() const {
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
    #endif
};

template <typename Real>
PieceWisePoly<Real> natural_cubic_spline(const arma::Col<Real>& x, const arma::Mat<Real>&y, const std::string& extrapolation="boundary", Real val=0) {
    u_long dim = y.n_cols;
    PieceWisePoly<Real> out(extrapolation, val, dim);
    
    if (x.empty()) {
        throw std::invalid_argument("natural_cubic_spline() error: x is empty.");
    }
    if (x.n_elem != y.n_rows) {
        throw std::invalid_argument("natural_cubic_spline() error: dimension mismatch, x.n_elem (=" + std::to_string(x.n_elem) + ") != y.n_rows (=" + std::to_string(y.n_elem) + ").");
    }

    u_long n = x.n_elem - 1;

    arma::SpMat<Real> A(n+1,n+1);
    arma::Mat<Real> RHS = arma::zeros<arma::Mat<Real>>(n+1, dim);
    arma::Mat<Real> b = arma::zeros<arma::Mat<Real>>(n, dim);
    arma::Mat<Real> d = arma::zeros<arma::Mat<Real>>(n, dim);
    arma::uvec I = arma::sort_index(x);
    arma::Col<Real> _x = x(I);
    for (u_long i=0; i < n; ++i) {
        if (_x(i+1) - _x(i) < std::abs(_x(i))*std::numeric_limits<Real>::epsilon) throw std::runtime_error("natural_cubic_spline() error: atleast two elements in x are repeating.");
    }
    arma::Mat<Real> _y = y.rows(I);

    arma::Col<Real> h = arma::diff<arma::Col<Real>>(_x);

    arma::Col<Real> subD = h;
    arma::Col<Real> supD = arma::zeros<arma::Col<Real>>(n);
    arma::Col<Real> mainD = arma::zeros<arma::Col<Real>>(n+1);

    subD(n-1) = 0;
    mainD(0) = 1;
    mainD(n) = 1;
    supD(0) = 0;

    for (u_long i=1; i < n; ++i) {     
        mainD(i) = 2 * (h(i) + h(i-1));
        supD(i) = h(i);

        RHS.row(i) = 3 * (_y.row(i+1) - _y.row(i))/h(i) - 3 * (_y.row(i) - _y.row(i-1))/h(i-1);
    }

    A.diag(-1) = subD;
    A.diag()   = mainD;
    A.diag(1)  = supD;

    arma::Mat<Real> c = arma::spsolve(A,RHS);

    for (u_long i=0; i < n; ++i) {
        b.row(i) = (_y.row(i+1) - _y.row(i))/h(i) - h(i)*(2*c.row(i) + c.row(i+1))/3;
        d.row(i) = (c.row(i+1) - c.row(i))/(3*h(i));
    }
    c = c.rows(0,n-1);

    for (u_long i=0; i < n; ++i) {
        arma::Mat<Real> p(4, dim);
        // translate from p(x - x[i]) to  p(z) for z = 2*(x - x[i])/(x[i+1] - x[i]) - 1
        p.row(0) = 0.125 * std::pow(h(i),3) * d.row(i); // h^3 d / 8
        p.row(1) = 0.125 * std::pow(h(i),2) * (2*c.row(i) + 3*h(i)*d.row(i)); // (2c + 3h*d)* h^2 / 8
        p.row(2) = 0.125 * h(i) * (4*b.row(i) + 4*h(i)*c.row(i) + 3*std::pow(h(i),2)*d.row(i)); // (4b +4h*c + 3*h^2*d) * h / 8
        p.row(3) = _y.row(i) + 0.125 * h(i) * (4*b.row(i) + 2*h(i)*c.row(i) + std::pow(h(i),2)*d.row(i)); // y + (4b + 2hc + h^2*d) * h / 8
        out.push(Polynomial<Real>(std::move(p), _x(i), _x(i+1)));
    }
    return out;
}

template <typename Real>
PieceWisePoly<Real> hermite_cubic_spline(const arma::Col<Real>& x, const arma::Mat<Real>& y, const arma::Mat<Real>& yp, const std::string& extrapolation="linear", Real val=0) {
    u_long dim = y.n_cols;
    PieceWisePoly out(extrapolation, val, dim);
    
    if (x.empty()) {
        throw std::invalid_argument("hermite_cubic_spline() error: x is empty.");
    }
    if (x.n_elem != y.n_rows) {
        throw std::invalid_argument("hermite_cubic_spline() error: dimension mismatch, x.n_elem (=" + std::to_string(x.n_elem) + ") != y.n_rows (=" + std::to_string(y.n_elem) + ").");
    }

    u_long n = x.n_elem;
    arma::uvec I = arma::sort_index(x);
    arma::Col<Real> _x = x(I);
    for (u_long i=0; i < n; ++i) {
        if (_x(i+1) - _x(i) < std::abs(_x(i+1))*arma::datum::eps) throw std::runtime_error("hermite_cubic_spline() error: atleast two elements in x are within epsilon of each other.");
    }
    arma::Mat<Real> _y = y.rows(I);
    arma::Mat<Real> _dy = yp.rows(I);

    for (u_long i=0; i < n-1; ++i) {
        arma::Mat<Real> p(4, dim);
        p.row(0) = 0.25 * ( _dy.row(i) + _dy.row(i+1) +   _y.row(i) -   _y.row(i+1));
        p.row(1) = 0.25 * (-_dy.row(i) + _dy.row(i+1));
        p.row(2) = 0.25 * (-_dy.row(i) - _dy.row(i+1) - 3*_y.row(i) + 3*_y.row(i+1));
        p.row(3) = 0.25 * ( _dy.row(i) - _dy.row(i+1) + 2*_y.row(i) + 2*_y.row(i+1));
        out.push(Polynomial<Real>(std::move(p), _x(i), _x(i+1)));
    }
    return out;
}

#ifdef NUMERICS_ODE_HPP
template <typename Real>
PieceWisePoly<Real> hermite_cubic_spline(const arma::Col<Real>& x, const arma::Mat<Real>& y, const std::string& extrapolation="linear", Real val=0) {
    arma::SpMat<Real> D;
    arma::uvec I = arma::sort_index(x);
    arma::Col<Real> _x = x(I);
    ode::diffmat(D, _x);
    arma::Mat<Real> _y = y.rows(I);
    arma::Mat<Real> yp = D * _y;
    return hermite_cubic_spline<Real>(_x, _y, _yp, extrapolation, val);
}
#endif // NUMERICS_ODE_HPP

#endif // NUMERICS_INTERPOLATION_POLYNOMIAL_HPP

}
#endif // NUMERICS_WITH_ARMA

#endif // NUMERICS_INTERPOLATION_PWFUNCTION_HPP