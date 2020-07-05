#include <numerics.hpp>

/* fsolve(f, x, max_iter) : Broyden's method for local root finding of nonlinear system of equations
 * --- f : f(x) = 0 function for finding roots of.
 * --- x : guess for root, also where root will be stored.
 * --- max_iter : maximum number of iterations after which method will stop regardless of convergence */
void numerics::broyd::fsolve(const std::function<arma::vec(const arma::vec&)>& f,
                            arma::vec& x,
                            int max_iter) {
    if (max_iter <= 0) {
        if (max_iterations <= 0) max_iterations = 100;
    } else max_iterations = max_iter;

    arma::vec F1,dx,y;
    
    arma::vec F = f(x);
    arma::mat J = approx_jacobian(f,x);

    uint k = 0;

    do {
        if (k >= max_iterations) {
            exit_flag = 1;
            num_iter += k;
            return;
        }

        dx = arma::solve(J,-F);
        x += dx;

        F1 = f(x);
        if (F1.has_nan() || F1.has_inf()) {
            exit_flag = 2;
            num_iter += k;
            return;
        }

        y = (F1 - F);
        J += (y - J*dx)*dx.t() / arma::dot(dx,dx);
        F = F1;

        if (F.has_nan() || F.has_inf() || J.has_nan() || J.has_inf()) {
            F = f(x);
            J = approx_jacobian(f,x);
        }
        k++;
    } while (arma::norm(F,"inf") > tol);
    num_iter += k;
    exit_flag = 0;
}

/* fsolve(f, x, max_iter) : Broyden's method for local root finding of nonlinear system of equations
 * --- f : f(x) = 0 function for finding roots of.
 * --- jacobian : J(x) jacobian of f(x).
 * --- x : guess for root, also where root will be stored.
 * --- max_iter : maximum number of iterations after which method will stop regardless of convergence */
void numerics::broyd::fsolve(const std::function<arma::vec(const arma::vec&)>& f,
                            const std::function<arma::mat(const arma::vec&)>& jacobian,
                            arma::vec& x, int max_iter) {
    if (max_iter <= 0) {
        if (max_iterations <= 0) max_iterations = 100;
    } else max_iterations = max_iter;

    arma::mat J;
    arma::vec F,F1,dx,y;
    
    F = f(x);
    J = jacobian(x);

    uint k = 0;

    do {
        if (k >= max_iterations) {
            exit_flag = 1;
            num_iter += k;
            return;
        }

        dx = arma::solve(J,-F);
        x += dx;

        F1 = f(x);
        if (F1.has_nan() || F1.has_inf()) {
            exit_flag = 2;
            num_iter += k;
            return;
        }

        y = (F1 - F);
        J += (y - J*dx)*dx.t() / arma::dot(dx,dx);
        F = F1;

        if (F.has_nan() || F.has_inf() || J.has_nan() || J.has_inf()) {
            F = f(x);
            J = jacobian(x);
        }
        k++;
    } while (arma::norm(F,"inf") > tol);
    num_iter += k;
    exit_flag = 0;
}