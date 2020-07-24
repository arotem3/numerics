#include <numerics.hpp>

double numerics::optimization::fminbnd(const std::function<double(double)>& f, double a, double b, double tol) {
    if (b <= a) {
        throw std::invalid_argument("invalid interval; a must be less than b but (a=" + std::to_string(a) + ", b=" + std::to_string(b) + ").");
    }
    if (tol <= 0) throw std::invalid_argument("fminbnd(): require tol (=" + std::to_string(tol) + ") > 0");
    double x,u,v,w,fu,fv,fw,fx;
    double c,d,e,m,p,q,r;

    c = (3 - std::sqrt(5))/2;
    x = a+c*(b-a); e=0; v = x; w = x;
    fx = f(x); fv = fx; fw = fx;

    while (std::max(x-a,b-x) > 2*tol) {
        m = (a+b)/2;
        p=0;q=0;r=0;
        bool pinterp_succ = false;
        if (std::abs(e) > tol) { // attempt parabolic interpolation step
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v)*q - (x-w)*r;
            q = 2*(q-r);
            if (q >= 0) p = -p;
            else q = -q;
            r = e;
            e = d;
            if ((std::abs(p)<std::abs(0.5*q*r)) && (p>q*(a-x)) && (p<q*(b-x))) { // parabolic interpolation step
                pinterp_succ = true;
                d = p/q;
                u = x+d;
                if (std::min(u-a,b-u) < 2*tol) { // we do not wish to evaluate the function at the end points
                    d = (x>=m) ? (tol) : (-tol);
                }
            }
        }
        if (!pinterp_succ) { // use golden section step
            if (x < m) e = b-x;
            else e = a-x;
            d = c*e;
        }


        if (std::abs(d) >= tol) u = x+d; // we do not want to evaluate the function again within tol of x.
        else if (d > 0) u = x+tol;
        else u = x-tol;

        fu = f(u);

        if (fu < fx) { // update points of interest
            if (u < x) b = x;
            else a = x;
            v=w; fv=fw;
            w=x; fw=fx;
            x=u; fx=fu;
        } else {
            if (u < x) a = u;
            else b = u;
            if ((fu <= fw)||(w==x)) {
                v=w; fv=fw;
                w=u; fw=fu;
            }
            else if ((fu <= fv)||(v==x)||(v==w)) {
                v=u; fv=fu;
            }
        }
    }
    return x;
}

double numerics::optimization::fminsearch(const std::function<double(double)>& f, double x0, double alpha) {
    uint best, worst;
    double tol = (std::abs(x0)<2e-8) ? (1e-8) : ((1e-8)*std::abs(x0));
    double R=1.0, E=2.0, Co=0.5, Ci=0.5;
    double xr, fr, xe, fe, xc, fc;
    double x[2], fx[2];

    if (alpha <= 0) alpha = 5*tol;

    x[0]=x0;
    x[1]=x0+alpha;
    fx[0]=f(x[0]);
    fx[1]=f(x[1]);

    while (std::abs(x[1] - x[0]) > 2*tol) {
        best = (fx[0] < fx[1]) ? (0) : (1);
        worst = not best;
        // attempt reflection step
        xr = (1+R)*x[best] - R*x[worst];
        fr = f(xr);
        if (fx[best] < fr && fr < fx[worst]) { // the reflection is better that worst but not as good as best, so we replace worst with the reflection
            x[worst] = xr;
            fx[worst] = fr;
        } else if (fr < fx[best]) { // the reflection is better than the best, so we try an even bigger step size
            xe = (1+E)*x[best] - E*x[worst];
            fe = f(xe);
            if (fe < fr) { // the expansion was successful
                x[worst] = xe;
                fx[worst] = fe;
            } else { // the expansion was not successful
                x[worst] = xr;
                fx[worst] = fr;
            }
        } else { // the reflection step was worse than the worst, so we take a smaller step size
            xc = (1+Co)*x[best] - Co*x[worst];
            fc = f(xc);
            if (fc < fr) { // contraction is better than the reflection so we keep it.
                x[worst] = xc;
                fx[worst] = fc;
            } else { // the contraction is worse so we replace worst with a point closer to best
                x[worst] = (1-Ci)*x[best] + Ci*x[worst];
                fx[worst] = f(x[worst]);
            }
        }
    }
    if (fx[0] < fx[1]) return x[0];
    else return x[1];
}