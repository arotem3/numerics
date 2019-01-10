#include "numerics.hpp"

//--- finds local root of single variable nonlinear functions using newton's method ---//
//----- f  : function to find root of -------------------------------------------------//
//----- df : derivative of f ----------------------------------------------------------//
//----- x0 : point near the root ------------------------------------------------------//
double numerics::newton(const dfunc& f, const dfunc& df, double x, double err) {
    err = std::abs(err); if (err <= 0) err = 1e-12;
    int max_iter = 100;
    
    double s;
    short k = 0;
    do {
        if (k >= max_iter) { // too many iterations
            std::cerr <<  "newton() failed: too many iterations needed to converge." << std::endl
                      << "returing current best estimate."
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "|f(x)| = " << std::abs(f(x)) << " > tolerance" << std::endl << std::endl;
            return NAN;
        }
        s = -f(x)/df(x);
        x += s;
        k++;
    } while (std::abs(s) > err);
    return x;
}

//--- secant methods for finding roots of single variable functions ---//
//--------- for added efficency we attempt ----------------------------//
//--------- to bracket the root with an auxilary point ----------------//
//----- f  : function to find root of ---------------------------------//
//----- x0 : point near the root --------------------------------------//
double numerics::secant(const dfunc& f, double a, double b) {
    double tol = 1e-8; int max_iter = 100;

    double fa = f(a), fb = f(b);
    int k = 2;
    if (std::abs(fa) < tol) return a;
    if (std::abs(fb) < tol) return b;
    
    double c = (a+b)/2;
    double fc = f(c); k++;

    while (std::abs(fc) > tol && std::abs(b-a) > tol) {
        if (k >= max_iter) { // too many iterations
            std::cerr << "secant() error: could not converge within " << max_iter << " function evaluations." << std::endl
                      << "\treturing current best estimate."
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "|f(x)| = " << std::abs(fc) << " > tolerance" << std::endl << std::endl;
            break;
        }
        double v;

        if (fc > 0) v = c - fc*(c-a)/(fc-fa);
        else v = b - fb*(b-c)/(fb-fc);
        
        double fv = f(v); k++;
        
        if (fv < 0) {
            a = c;
            fa = fc;
        } else {
            b = c;
            fb = fc;
        }
        c = v;
        fc = fv;
    }
    return c;
}

//--- bisection method for finding roots of single variable functions ---//
//----- f  : function to find root of -----------------------------------//
//----- x1 : lower bound of search interval -----------------------------//
//----- x2 : upper bound of search interval -----------------------------//
double numerics::bisect(const dfunc& f, double a, double b, double tol) {
    tol = std::abs(tol); int max_iter = 100;
    short k = 2;

    double fa = f(a), fb = f(b);

    if (std::abs(fa) < tol ) {
        return a;
    } else if (std::abs(fb) < tol ) {
        return b;
    }

    if (fa*fb > 0) { // bracketing error
        std::cerr << "bisect() error: provided points do not bracket a simple root." << std::endl;
        return NAN;
    }

    double c, fc;

    do {
        if (k >= max_iter) { // too many iterations
            std::cerr << "bisect() error: could not converge within " << max_iter << " function evaluations." << std::endl
                      << "returing current best estimate."
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "|f(x)| = " << std::abs(fc) << " > tolerance" << std::endl << std::endl;
            break;
        }
        c = (a+b)/2;
        fc = f(c); k++;
        if (std::abs(fc) < tol) break;
        if (fc*fa < 0) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    } while (std::abs(fc) > tol && std::abs(b-a) > tol);

    return (a + b)/2;
}

//--- narrows down to root using bisection method then converges using secant method ---//
//----- f  : function to find root of --------------------------------------------------//
//----- x1 : lower bound of search interval --------------------------------------------//
//----- x2 : upper bound of search interval --------------------------------------------//
double numerics::fzero(const dfunc& f, double a, double b) {
    double tol = 1e-10; int max_iter = 100;

    double fa = f(a), fb = f(b);
    int k = 2;
    if (std::abs(fa) < tol) return a;
    if (std::abs(fb) < tol) return b;

    if (fa*fb > 0) {
        std::cerr << "fzero() error: provided points do not bracket a simple root." << std::endl;
        return NAN;
    }
    
    double c = (a+b)/2;
    double fc = f(c); k++;

    while (std::abs(fc) > tol && std::abs(b-a) > tol) {
        if (k >= max_iter) { // error
            std::cerr << "fzero() error: could not converge within " << max_iter << " function evaluations." << std::endl
                      << "returing current best estimate."
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "|f(x)| = " << std::abs(fc) << " > tolerance" << std::endl << std::endl;
            break;
        }
        double v;
        if (std::abs(fa-fc) > tol || std::abs(fb-fc) > tol) { // inverse quadratic
            v = (a*fb*fc)/((fa-fb)*(fa-fc)) + (b*fa*fc)/((fb-fa)*(fb-fc)) + (c*fa*fb)/((fc-fa)*(fc-fb));
        } else { // secant
            if (fc > 0) v = c - fc*(c-a)/(fc-fa);
            else v = b - fb*(b-c)/(fb-fc);
        }
        double fv = f(v); k++;
        if (fv < 0) {
            a = c;
            fa = fc;
        } else {
            b = c;
            fb = fc;
        }
        c = v;
        fc = fv;
    }
    return c;
}