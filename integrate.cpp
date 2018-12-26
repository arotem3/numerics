#include "numerics.hpp"

double Simp(const numerics::dfunc&, double, double, int);
double trap(const numerics::dfunc&, double, double, int);

double numerics::integrate(const dfunc& f, double a, double b, integrator i, double err) {
    double s = 0;
    if (i == SIMPSON) {
        s = Sintegrate(f,a,b,err);
    } else if (i == TRAPEZOID) {
        s = Tintegrate(f,a,b,err);
    } else if (i == LOBATTO) {
        s = Lintegrate(f,a,b,err);
    } else {
        std::cerr << "integrate() error: invalid input selection." << std::endl;
    }
    return s;
}

//--- adaptive quadrature method for single variable functions ---//
//----- f  : f(x) to integrate -----------------------------------//
//----- a  : lower bound of integral -----------------------------//
//----- b  : upper bound of integral -----------------------------//
//----- err: upper bound on error of integral --------------------//
double numerics::Sintegrate(const dfunc& f, double a, double b, double err) { //adaptive method for integration
    err = std::abs(err);
    double S1 = Simp(f,a,b,2);
    double S2 = Simp(f,a,b,4);
    double s;
    
    if (std::abs(S2 - S1) < 15*err) { //results from comparing 1 and 2 intervals of simpson's rule
        s = S2;
    }
    else {
        s = Sintegrate(f, a, (a+b)/2, err/2) + Sintegrate(f, (a+b)/2, b, err/2);
    }
    
    return s;
}

//--- adaptive quadrature using trapezoid rule: ideal for periodic or discontinuous functions ---//
//----- f  : f(x) to integrate ------------------------------------------------------------------//
//----- a  : lower bound of integral ------------------------------------------------------------//
//----- b  : upper bound of integral ------------------------------------------------------------//
//----- err: upper bound on error of integral ---------------------------------------------------//
double numerics::Tintegrate(const dfunc& f, double a, double b, double err) {
    err = std::abs(err);
    double T1 = trap(f,a,b,2);
    double T2 = trap(f, a,b,4);
    double t;
    if (std::abs(T2 - T1) < 3*err) {
        t = T2;
    } else {
        t = Tintegrate(f, a, (a+b)/2, err/2) + Tintegrate(f, (a+b)/2, b, err/2);
    }
    return t;
}

//--- adaptive Gauss-Lobato quadrature algorithm: ideal for smooth, easy to evaluate functions ---//
//----- f  : f(x) to integrate -------------------------------------------------------------------//
//----- a  : lower bound -------------------------------------------------------------------------//
//----- b  : upper bound -------------------------------------------------------------------------//
//----- err: upper bound on error ----------------------------------------------------------------//
double numerics::Lintegrate(const dfunc& f, double a, double b, double err) {
    err = std::abs(err);
    double bma = (b-a)/2;
    double bpa = (b+a)/2;

    double sum4(0), sum7(0);
    for (int i(0); i < 7; i++) {
        if (i < 4) {
            sum4 += W4[i] * f(bma * X4[i] + bpa);
        }
        sum7 += W7[i] * f(bma * X7[i] + bpa);
    }
    sum4 *= bma;
    sum7 *= bma;

    if (std::abs(sum4 - sum7) < err) return sum4;
    else return Lintegrate(f,a,bpa,err/2) + Lintegrate(f,bpa,b,err/2);
}

//--- simpson's method for integration ---//
//----- f  : f(x) to integrate -----------//
//----- a  : lower bound of integral -----//
//----- b  : upper bound of integral -----//
//----- n  : number of subintervals ------//
double Simp(const numerics::dfunc& f, double a, double b, int n) { //simpson's rule for integration
    double h = (b - a)/n;
    double s = f(a) + f(b);
    for (int i(2); i<=n; ++i) {
        if (i%2 == 0) {
            s += 4 * f(a + (i - 1)*h);
        }
        else {
            s += 2 * f(a + (i - 1)*h);
        }
    }
    s *= h/3;
    return s;
}

//--- trapezoid method for integration ---//
double trap(const numerics::dfunc& f, double a, double b, int n) {
    double h = (b-a)/n;
    double t = 0;
    double temp;
    for (int i(0); i < n; ++i) {
        temp = f(a + i*h) + f(a + (i+1)*h);
        temp *= h/2;
        t += temp;
    }
    return t;
}