#include <numerics.hpp>

double numerics::simpson_integral(std::map<double,double>& fvals, const std::function<double(double)>& f, double a, double b, double tol) {
    if (tol <= 0) throw std::invalid_argument("require tol (=" + std::to_string(tol) + ") > 0");
    if (b <= a) throw std::invalid_argument("(" + std::to_string(a) + ", " + std::to_string(b) + ") does not define a valid interval");

    double integral = 0;
    double l, c, r, mid1, mid2, h;
    c = (a + b) / 2;
    l = (a + c) / 2;
    r = (c + b) / 2;

    std::queue<std::vector<double>> q; q.push({a,l,c,r,b});
    std::queue<double> tq; tq.push(tol);
    while (not q.empty()) {
        for (double val : q.front()) {
            if (fvals.count(val) == 0) fvals[val] = f(val);
        }
        a = q.front().at(0);
        l = q.front().at(1);
        c = q.front().at(2);
        r = q.front().at(3);
        b = q.front().at(4);
        q.pop();
        tol = tq.front(); tq.pop();
        
        h = (b - a) / 2;
        double s1 = (fvals[a] + 4*fvals[c] + fvals[b]) * h / 3;
        double s2 = (fvals[a] + 4*fvals[l] + 2*fvals[c] + 4*fvals[r] + fvals[b]) * h / 6;
        if (std::abs(s2 - s1) < 15*tol) integral += s2;
        else {
            mid1 = (a + l)/2; mid2 = (l + c)/2;
            q.push({a,mid1,l,mid2,c});
            tq.push(tol/2);

            mid1 = (c + r)/2; mid2 = (r + b)/2;
            q.push({c,mid1,r,mid2,b});
            tq.push(tol/2);
        }
    }
    return integral;
}

// double numerics::simpson_integral(const std::function<double(double)>& f, double a, double b, std::map<double,double>& fmap, double tol) {
//     double l, c, r;
//     c = (a+b)/2;
//     l = (a+c)/2;
//     r = (c+b)/2;

//     if (fmap.count(a)==0) fmap[a] = f(a);
//     if (fmap.count(l)==0) fmap[l] = f(l);
//     if (fmap.count(c)==0) fmap[c] = f(c);
//     if (fmap.count(r)==0) fmap[r] = f(r);
//     if (fmap.count(b)==0) fmap[b] = f(b);

//     double h = (b-a)/2;

//     double s1 = (fmap[a] + 4*fmap[c] + fmap[b])*h/3;
//     double s2 = (fmap[a] + 4*fmap[l] + 2*fmap[c] + 4*fmap[r] + fmap[b])*h/6;

//     if (std::abs(s2-s1) < 15*tol) return s2;
//     else return simpson_integral(f,a,c,fmap,tol/2) + simpson_integral(f,c,b,fmap,tol/2);
// }

double numerics::simpson_integral(const std::function<double(double)>& f, double a, double b, double tol) {
    std::map<double,double> fmap;
    return simpson_integral(fmap, f, a, b, tol);
}