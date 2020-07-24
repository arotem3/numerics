#include "numerics.hpp"

double numerics::optimization::newton_1d(const std::function<double(double)>& f, const std::function<double(double)>& df, double x, double err) {
    err = std::abs(err); if (err <= 0) err = 1e-12;
    int max_iter = 100;
    
    double s;
    long long k = 0;
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

double numerics::optimization::secant(const std::function<double(double)>& f, double a, double b, double tol) {
    int max_iter = 100;

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

double numerics::optimization::bisect(const std::function<double(double)>& f, double a, double b, double tol) {
    tol = std::abs(tol);
    long long k = 2;

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

double numerics::optimization::fzero(const std::function<double(double)>& f, double a, double b, double tol) {
    int max_iter = std::min(std::pow(std::log2((b-a)/tol)+1,2), 1.0e2); // will nearly never happen

    double c, d, e, fa, fb, fc, m=0, s=0, p=0, q=0, r=0, t, eps = std::numeric_limits<double>::epsilon();
    int k=0;
    fa = f(a); k++;
    fb = f(b); k++;
    if (std::abs(fa) == 0) return a;
    if (std::abs(fb) == 0) return b;

    if (fa*fb > 0) {
        std::cerr << "fzero() error: provided points do not bracket a simple root." << std::endl;
        return NAN;
    }
    
    c = a; fc = fa; d = b-a; e = d;

    while (true) {
        if (std::abs(fc) < std::abs(fb)) {
             a =  b;  b =  c;  c =  a;
            fa = fb; fb = fc; fc = fa;
        }
        m = (c-b)/2;
        t = 2*std::abs(b)*eps + tol;
        if (std::abs(m) < t || fb == 0) break; // convergence criteria
        if (k >= max_iter) {
            std::cerr << "fzero() error: could not converge within " << max_iter << " function evaluations (the estimated neccessary ammount).\n"
                      << "returing current best estimate.\n"
                      << "!!!---not necessarily a good estimate---!!!\n"
                      << "|dx| = " << std::abs(m) << " > " << tol << "\n";
            break;
        }

        if (std::abs(e) < t || std::abs(fa) < std::abs(fb)) { // bisection
            d = m; e = m;
        } else {
            s = fb/fa;
            if (a == c) { // secant
                p = 2*m*s;
                q = 1 - s;
            } else { // inverse quadratic
                q = fa/fc;
                r = fb/fc;
                p = s*(2*m*q*(q-r)-(b-a)*(r-1));
                q = (q-1)*(r-1)*(s-1);
            }

            if (p > 0) q = -q;
            else p = -p;

            s = e; e = d;

            if (2*p < 3*m*q - std::abs(t*q) && p < std::abs(0.5*s*q)) d = p/q;
            else {
                d = m; e = m;
            }
        }
        a = b; fa = fb;

        if (std::abs(d) > t) b += d;
        else if (m > 0) b += t;
        else b -= t;

        fb = f(b); k++;

        if (fb*fc > 0) {
            c = a; fc = fa;
            e = b-a; d = e;
        }
    }
    return b;
}