#include <numerics.hpp>

/* find_fixed_point(f, x, max_iter) : anderson mixing fixed point iteration. Finds solutions of the problem x = f(x).
 * --- f : vector function of x = f(x).
 * --- x : initial guess and solution output.
 * --- max_iter : maximum number of iterations allowed. */
void numerics::mix_fpi::find_fixed_point(std::function<arma::vec(const arma::vec&)> f,
                              arma::vec& x,
                              int max_iter) {
    if (max_iter <= 0) {
        if (max_iterations <= 0) max_iterations = 100;
    } else max_iterations = max_iter;

    int n = x.n_elem;
    numerics::numerics_private_utility::cyc_queue F(n, steps_to_remember), X(n, steps_to_remember);
    arma::mat FF;
    arma::vec v, b = arma::zeros(n+1);
    b(n) = 1;

    uint k = 0;
    do {
        if (k > max_iterations) {
            exit_flag = 1;
            num_iter += k;
            return;
        }

        v = f(x);
        F.push(v);
        X.push(x);
        int m = std::min(k+1, steps_to_remember);
        FF = arma::ones(n+1,m);
        FF.rows(0,n-1) = X.data() - F.data();

        x = F.data() * arma::solve(FF,b);
        k++;
    } while (arma::norm(v - x,"inf") > tol);
    num_iter += k;

    exit_flag = 0;
}