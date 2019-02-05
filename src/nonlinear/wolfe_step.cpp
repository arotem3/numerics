#include "numerics.hpp"

/* WOLFE_STEP : rough step size approximator for quasi-newton methods based on strong wolfe conditions.
 * --- obj_func : objective function.
 * --- f : gradient function.
 * --- x : current guess.
 * --- p : search direction.
 * --- c1 : wolfe constant 1.
 * --- c2 : wolfe constant 2.
 * --- b : line minimization constant. */
double numerics::wolfe_step(const vec_dfunc& obj_func, const vector_func& f, const arma::vec& x, const arma::vec& p, double c1, double c2, double b) {
    double alpha = 1;
    int k = 0;
    while (true) {
        if (k > 100) break;
        double pfa = arma::dot(p, f(x + alpha*x));
        bool cond1 = obj_func(x + alpha*p) <= obj_func(x) + c1*alpha*pfa;
        bool cond2 = std::abs(arma::dot(p, f(x + alpha*p))) <= c2*std::abs(arma::dot(p, f(x)));
        if (cond1 && cond2) break;
        else if (obj_func(x + b*alpha*p) < obj_func(x + alpha*p)) alpha *= b;
        else alpha /= b;
        k++;
    }
    return alpha;
}