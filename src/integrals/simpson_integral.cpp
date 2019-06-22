#include <numerics.hpp>

/* simpson_integral(f, a, b, fmap, tol) : adaptive simpson's method, generally efficient.
 * --- f : function to integrate.
 * --- a,b : interval [a,b] to evaluate integral over.
 * --- fmap : if fmap is provided, all function evaluations will be stored here.
 * --- tol : error tolerance, i.e. stopping criterion */
double numerics::simpson_integral(const std::function<double(double)>& f, double a, double b, std::map<double,double>& fmap, double tol) {
    double l, c, r;
    c = (a+b)/2;
    l = (a+c)/2;
    r = (c+b)/2;

    if (fmap.count(a)==0) fmap[a] = f(a);
    if (fmap.count(l)==0) fmap[l] = f(l);
    if (fmap.count(c)==0) fmap[c] = f(c);
    if (fmap.count(r)==0) fmap[r] = f(r);
    if (fmap.count(b)==0) fmap[b] = f(b);

    double h = (b-a)/2;

    double s1 = (fmap[a] + 4*fmap[c] + fmap[b])*h/3;
    double s2 = (fmap[a] + 4*fmap[l] + 2*fmap[c] + 4*fmap[r] + fmap[b])*h/6;

    if (std::abs(s2-s1) < 15*tol) return s2;
    else return simpson_integral(f,a,c,fmap,tol/2) + simpson_integral(f,c,b,fmap,tol/2);
}

/* simpson_integral(f, a, b, tol) : adaptive simpson's method, generally efficient.
 * --- f : function to integrate.
 * --- a,b : interval [a,b] to evaluate integral over.
 * --- tol : error tolerance, i.e. stopping criterion */
double numerics::simpson_integral(const std::function<double(double)>& f, double a, double b, double tol) {
    std::map<double,double> fmap;
    return simpson_integral(f, a, b, fmap, tol);
}