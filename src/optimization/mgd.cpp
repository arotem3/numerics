#include <numerics.hpp>

/* minimize(grad_f, x, max_iter) : momentum gradient descent.
 * --- f : gradient function.
 * --- x : initial guess.
 * --- max_iter : maximum number of iterations allowed. */
void numerics::mgd::minimize(const std::function<arma::vec(const arma::vec&)>& grad_f, arma::vec& x, int max_iter) {
    if (max_iter <= 0) {
        if (max_iterations <= 0) max_iterations = 100;
    } else max_iterations = max_iter;

    bool minimize_line = (step_size <= 0);
    arma::vec p,g;
    p = grad_f(x);
    double r = arma::norm(p,"inf");
    double alpha = step_size;
    if (minimize_line) alpha = numerics::line_min(
        [&p,&x,&grad_f,r](double a) -> double {
            arma::vec q = (-1.0/r)*p;
            return arma::dot( q, grad_f(x+a*q) );
        }
    );
    x += (-alpha/r)*p;

    uint k = 1;
    do {
        if (k >= max_iterations) {
            exit_flag = 1;
            num_iter += k;
            return;
        }
        g = grad_f(x);

        if (g.has_nan() || g.has_inf()) {
            exit_flag = 2;
            num_iter += k;
            return;
        }

        p = damping_param*p + g;
        r = arma::norm(p,"inf");
        if (minimize_line) alpha = numerics::line_min(
            [&p,&x,&grad_f,r](double a) -> double {
                arma::vec q = (-1.0/r)*p;
                return arma::dot( q, grad_f(x+a*q) );
            }
        );
        x += (-alpha/r)*p;
        k++;
    } while (arma::norm(g,"inf") > tol);
    num_iter += k;
    exit_flag = 0;
}