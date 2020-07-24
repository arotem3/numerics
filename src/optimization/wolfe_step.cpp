#include "numerics.hpp"

double numerics::optimization::wolfe_step(const dFunc& f, const VecFunc& grad_f, const arma::vec& x, const arma::vec& p, double c1, double c2) {
    auto g = [&](double a) -> double {
        return f(x + a*a*p);
    };
    uint best, worst, n_iter=0;
    double R=1.0, E=2.0, Co=0.5, Ci=0.5;
    double ar, fr, ae, fe, ac, fc;
    double a[2], fa[2];

    a[0]=0;
    a[1]=1;
    fa[0]=g(a[0]);
    fa[1]=g(a[1]);

    while (true) {
        best = (fa[0] < fa[1]) ? (0) : (1);
        worst = not best;

        bool cond1 = g(a[best]) <= f(x) + c1*a[best]*fa[best];
        bool cond2 = std::abs(arma::dot(p, grad_f(x + a[best]*p))) <= c2*std::abs(arma::dot(p, grad_f(x)));
        if ((cond1 && cond2) || n_iter >= 100) break;

        // attempt reflection step
        ar = (1+R)*a[best] - R*a[worst];
        fr = g(ar);
        if (fa[best] < fr && fr < fa[worst]) { // the reflection is better that worst but not as good as best, so we replace worst with the reflection
            a[worst] = ar;
            fa[worst] = fr;
        } else if (fr < fa[best]) { // the reflection is better than the best, so we try an even bigger step size
            ae = (1+E)*a[best] - E*a[worst];
            fe = g(ae);
            if (fe < fr) { // the expansion was successful
                a[worst] = ae;
                fa[worst] = fe;
            } else { // the expansion was not successful
                a[worst] = ar;
                fa[worst] = fr;
            }
        } else { // the reflection step was worse than the worst, so we take a smaller step size
            ac = (1+Co)*a[best] - Co*a[worst];
            fc = g(ac);
            if (fc < fr) { // contraction is better than the reflection so we keep it.
                a[worst] = ac;
                fa[worst] = fc;
            } else { // the contraction is worse so we replace worst with a point closer to best
                a[worst] = (1-Ci)*a[best] + Ci*a[worst];
                fa[worst] = g(a[worst]);
            }
        }
        n_iter++;
    }
    if (fa[0] < fa[1]) return a[0]*a[0];
    else return a[1]*a[1];
}