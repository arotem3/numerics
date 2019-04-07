#include "numerics.hpp"

/* INTEGRATE : adaptive numerical intergration with user choice of algorithm
 * --- f : function to integrate.
 * --- a,b : interval [a,b] to evaluate integral over
 * --- i : choice of integrator {SIMPSON, TRAPEZOID, LOBATTO}
 * --- err : error tolerance, i.e. stopping criterion */
double numerics::integrate(const dfunc& f, double a, double b, integrator i, double err) {
    double s = 0;
    if (i == SIMPSON) {
        s = simpson_integral(f,a,b,err);
    } else {
        s = lobatto_integral(f,a,b,err);
    }
    return s;
}

/* SINTEGRATE : adaptive simpson's method, a generally good method
 * --- f : function to integrate.
 * --- a,b : interval [a,b] to evaluate integral over
 * --- err : error tolerance, i.e. stopping criterion */
struct tree {
    double I;
    double xL, xM, xR;
    double h;
    tree* left;
    tree* right;
    void compute_integral(std::map<double,double>& F) {
        I = ( F[xL] + 4*F[xM] + F[xR] ) * h/3;
    }
    void build_children(const numerics::dfunc& f, std::map<double,double>& F) {
        left = new tree;
        left->xL = xL;
        left->xM = (xL + xM)/2;
        left->xR = xM;
        left->h  = h/2;
        F[left->xM] = f(left->xM);
        left->compute_integral(F);
        
        right = new tree;
        right->xL = xM;
        right->xM = (xM + xR)/2;
        right->xR = xR;
        right->h  = h/2;
        F[right->xM] = f(right->xM);
        right->compute_integral(F);
    }
};

double eval_tree(tree* val, const numerics::dfunc& f, std::map<double,double>& F, double err) {
    double I1 = val->I;
    double I2 = val->left->I + val->right->I;
    if (std::abs(I1 - I2) < 15*err) return I1;
    else {
        err /= 2;
        val->left->build_children(f, F);
        val->right->build_children(f, F);
        return eval_tree(val->left, f, F, err) + eval_tree(val->right, f, F, err);
    }
}

double numerics::simpson_integral(const numerics::dfunc& f, double a, double b, double err) {
    tree* val = new tree;
    val->xL = a;
    val->xM = (a+b)/2;
    val->xR = b;
    val->h = (val->xR - val->xL)/2;

    std::map<double,double> F;
    F[val->xL] = f(val->xL);
    F[val->xM] = f(val->xM);
    F[val->xR] = f(val->xR);

    val->compute_integral(F);
    val->build_children(f, F);

    return eval_tree(val, f, F, err);
}

/* LINTEGRATE : adaptive gauss-Lobato's method,
 * highly accurate and fast for smooth functions
 * --- f : function to integrate.
 * --- a,b : interval [a,b] to evaluate integral over
 * --- err : error tolerance, i.e. stopping criterion */
double numerics::lobatto_integral(const dfunc& f, double a, double b, double err) {
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
    else return lobatto_integral(f,a,bpa,err/2) + lobatto_integral(f,bpa,b,err/2);
}