#include <numerics.hpp>

/* fsolve(f, jacobian, x, max_iter) : finds a local root of a multivariate nonlinear system of equations using newton's method.
 * --- f  : f(x) == 0, system of equations.
 * --- jacobian  : J(x) jacobian of system.
 * --- x : initial guess as to where the root, also where the root will be returned to.
 * --- max_iter : maximum number of iterations allowed. */
void numerics::newton::fsolve(std::function<arma::vec(const arma::vec&)> f,
                             std::function<arma::mat(const arma::vec&)> jacobian,
                             arma::vec& x,
                             int max_iter) {
    if (max_iter <= 0) {
        if (max_iterations <= 0) max_iterations = 100;
    } else max_iterations = max_iter;
    
    arma::vec F,dx;
    arma::mat J;
    uint k = 0;

    do {
        if (k > max_iterations) {
            exit_flag = 1;
            num_iter += k;
            return;
        }
        F = -f(x);
        J = jacobian(x);
        if (use_cgd) cgd(J,F,dx);
        else dx = arma::solve(J,F);
        x += dx;
        k++;
    } while ( arma::norm(dx, "inf") > tol );

    num_iter += k;
    exit_flag = 0;
}