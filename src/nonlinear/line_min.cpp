#include "numerics.hpp"

/* LINE_MIN : solves the line minimization sub-problem for local minimization
 * --- f : f(a) = p' * f(x + a*p) */
double numerics::line_min(const dfunc& f) {
    int max_iter_i = 100;
    int max_iter = 100;
    double tol = 1e-8;
    double interp_tol = 1e-8;

    double a = 0,
           b = 1,
           s = 0.75,
           t = 1.25,
           fb = f(b),
           c, fa, fc, v;
    int k = 1;
    
    // -- initialise bracket
    while (fb < 0) {
        if (k >= max_iter_i) break;
        double fb1 = f(s*b);
        if (fb1 < 0) { // s*b does not bracket a minimum
            double fb2 = f(t*b);
            if (fb2 > 0 || fb2 > fb1) { // t*b does brack a minimum or is closer to a min
                b *= t;
                fb = fb2;
                std::swap(s,t);
            } else { // s*b is closer than t*b to the minimum
                b *= s;
                fb = fb1;
            }
        } else {
            b *= s;
            fb = fb1;
        }
        k++;
    }
    fa = f(a);

    // -- begin root finding over bracket
    if (std::abs(fb) < tol) return b;
    c = (a+b)/2;
    fc = f(c);

    k = 1;
    while (std::abs(fc) > tol && std::abs(b-a) > tol) {
        if (k >= max_iter) break;
        if (std::abs(fa-fc) > interp_tol && std::abs(fb-fc) > interp_tol) { // inverse quadratic
            v = (a*fb*fc)/((fa-fb)*(fa-fc)) + (b*fa*fc)/((fb-fa)*(fb-fc)) + (c*fa*fb)/((fc-fa)*(fc-fb));
        } else { // secant
            if (fc > 0) v = c - fc*(c-a)/(fc-fa);
            else v = b - fb*(b-c)/(fb-fc);
        }
        double fv = f(v);
        if (fv < 0) {
            a = c;
            fa = fc;
        } else {
            b = c;
            fb = fc;
        }
        c = v;
        fc = fv;
        k++;
    }
    return c;
}