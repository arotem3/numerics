#include "numerics.hpp"

/* wolfe_step(f, grad_f, x, p, c1, c2, b) : rough step size approximator for quasi-newton methods based on strong wolfe conditions.
 * --- f : objective function.
 * --- grad_f : gradient function.
 * --- x : current guess.
 * --- p : search direction.
 * --- c1 : wolfe constant 1.
 * --- c2 : wolfe constant 2.
 * --- b : line minimization constant. */
double numerics::wolfe_step(const std::function<double(const arma::vec&)>& f,
                            const std::function<arma::vec(const arma::vec&)>& grad_f,
                            const arma::vec& x,
                            const arma::vec& p,
                            double c1, double c2, double b) {
    double alpha = 1;
    int k = 0;
    while (true) {
        if (k > 100) break;
        double pfa = arma::dot(p, grad_f(x + alpha*x));
        bool cond1 = f(x + alpha*p) <= f(x) + c1*alpha*pfa;
        bool cond2 = std::abs(arma::dot(p, grad_f(x + alpha*p))) <= c2*std::abs(arma::dot(p, grad_f(x)));
        if (cond1 && cond2) break;
        else if (f(x + b*alpha*p) < f(x + alpha*p)) alpha *= b;
        else alpha /= b;
        k++;
    }
    return alpha;
}