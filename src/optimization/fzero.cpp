#include "numerics.hpp"

double numerics::optimization::newton_1d(const std::function<double(double)>& f, const std::function<double(double)>& df, double x, double tol) {
    if (tol <= 0) throw std::invalid_argument("error bound should be strictly positive, but tol=" + std::to_string(tol));
    int max_iter = 100;
    
    double s=tol/2;
    u_long k = 0;
    double fx, fp;
    do {
        if (k >= max_iter) { // too many iterations
            std::cerr << "newton_1d() failed: too many iterations needed to converge." << std::endl
                      << "returing current best estimate."
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "|f(x)| = " << std::abs(f(x)) << " > tolerance" << std::endl << std::endl;
            return x;
        }
        fx = f(x);
        fp = df(x);
        if (std::abs(fp) < tol/2) s *= -fx/(fx - f(x-s));
        else s = -fx/fp;
        x += s;
        k++;
    } while ((std::abs(fx) > tol) && (std::abs(s) > tol));
    return x;
}

double numerics::optimization::newton_1d(const std::function<double(double)>& f, double x, double tol) {
    auto df = [&](double u) -> double {
        return deriv(f, u, tol/2, true, 2);
    };
    return newton_1d(f, df, x, tol);
}

double numerics::optimization::secant(const std::function<double(double)>& f, double a, double b, double tol) {
    int max_iter = 100;

    double fa = f(a), fb = f(b);
    int k = 2;
    if (std::abs(fa) < tol) return a;
    if (std::abs(fb) < tol) return b;

    while (true) {
        if (k >= max_iter) { // too many iterations
            std::cerr << "secant() error: could not converge within " << max_iter << " function evaluations." << std::endl
                      << "\treturing current best estimate."
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "|f(x)| = " << std::min(std::abs(fa), std::abs(fb)) << " > tolerance" << std::endl << std::endl;
            return (std::abs(fa) < std::abs(fb)) ? a : b;
        }
        double c;

        double rel = std::max( std::max(std::abs<double>(fb), std::abs<double>(fa)), 1.0);
        if (std::abs(fa - fb) < 1e-10*rel) c = (a + b) / 2;
        else c = b - fb*(b-a)/(fb-fa);

        double fc = f(c); k++;

        if (std::abs(fc) < tol) return c;

        if (fa*fb < 0) {
            if (fb*fc < 0) {
                a = c; fa = fc;
            } else {
                b = c; fb = fc;
            }
        } else {
            if (std::abs(fa) < std::abs(fb)) {
                b = c; fb = fc;
            } else {
                a = c; fa = fc;
            }
        }

        rel = std::max( std::max(std::abs<double>(a), std::abs<double>(b)), 1.0);
        if (std::abs(a - b) < rel*tol) return (a+b)/2;
    }
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