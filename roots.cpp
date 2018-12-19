#include "numerics.hpp"

//--- finds local root of single variable nonlinear functions using newton's method ---//
//----- f  : function to find root of -------------------------------------------------//
//----- df : derivative of f ----------------------------------------------------------//
//----- x0 : point near the root ------------------------------------------------------//
double numerics::newton(std::function<double(double)> f, std::function<double(double)> df, double x0, double err) {
    auto p = [&f,&df](double x) -> double { return x - f(x)/df(x); };
    double x1 = x0;
    double x2 = p(x1);
    short k = 0;
    while (std::abs(x2-x1) > err) {
        if (k > 100) {
            std::cerr <<  "newton() failed: too many iterations needed to converge." << std::endl;
            return NAN;
        }
        x1 = x2;
        x2 = p(x2);
        k++;
    }
    return x2;
}

//--- secant methods for finding roots of single variable functions ---//
//----- f  : function to find root of ---------------------------------//
//----- x1 : point near the root --------------------------------------//
double numerics::secant(std::function<double(double)> f, double x0, double err) {
    auto df = [&](double x) -> double {
        return deriv(f,x,err);
    };
    return newton(f,df,x0,err);
}

//--- bisection method for finding roots of single variable functions ---//
//----- f  : function to find root of -----------------------------------//
//----- x1 : lower bound of search interval -----------------------------//
//----- x2 : upper bound of search interval -----------------------------//
double numerics::bisect(std::function<double(double)> f, double x1, double x2, double tol) {
    if (tol <= 0) tol = eps(x2); // check tolerance condition
    if (std::abs(f(x1)) < tol ) {
        return x1;
    } else if (std::abs(f(x2)) < tol ) {
        return x2;
    }

    while ( std::abs(x2 - x1) > tol ) {
        double p = (x2 + x1)/2;
        if ( std::abs(f(p)) < tol ) {
            return p;
        } else if (f(x1) * f(p) < 0) {
            x2 = p;
        } else {
            x1 = p;
        }
    }

    if ( std::abs(f(x2)) > tol ) { // error control
        std::cerr << "bisect() failed: there is likely no simple roots on the interval." << std::endl;
        return NAN;
    }
    return (x1 + x2)/2;
}


//--- narrows down to root using bisection method then converges using secant method ---//
//----- f  : function to find root of --------------------------------------------------//
//----- x1 : lower bound of search interval --------------------------------------------//
//----- x2 : upper bound of search interval --------------------------------------------//
double numerics::roots(std::function<double(double)> f, double x1, double x2) {
    if (std::abs(f(x1)) < eps(x1)) {
        return x1;
    } else if (std::abs(f(x2)) < eps(x2)) {
        return x2;
    }

    // ---  bisect to reduce problem
    double p;
    while (std::abs(x2 - x1) > 0.1) {
        p = (x2 + x1)/2;
        if ( std::abs(f(p)) < eps(p) ) {
            return p;
        } else if (f(x1) * f(p) < 0) {
            x2 = p;
        } else {
            x1 = p;
        }
    }

    // --- secant to the rest of the way
    p = (x2 + x1)/2;
    return secant(f,p,1e-10);
}