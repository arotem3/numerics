#include <numerics.hpp>

/* lbfgs_update(p,S,Y) : update function for limited memory BFGS
 * --- p : negative gradient, the search direction is also stored here
 * --- S : s_i = x_i - x_(i-1) 
 * --- Y : y_i = f_i - f_(i-1) */
void numerics::lbfgs::lbfgs_update(arma::vec& p,
                                   numerics_private_utility::cyc_queue& S,
                                   numerics_private_utility::cyc_queue& Y,
                                   const double& hdiag) {
    int k = S.length();

    arma::vec ro = arma::zeros(k);
    for (int i(0); i < k; ++i) {
        ro(i) = 1 / arma::dot(S(i),Y(i));
    }

    arma::vec q = p;
    arma::vec alpha = arma::zeros(k);
    
    for (int i(k-1); i >= 0; --i) {
        alpha(i) = ro(i) * arma::dot(S(i),q);
        q -= alpha(i) * Y(i);
    }

    arma::vec r = q * hdiag;

    for (int i(0); i < k; ++i) {
        double beta = ro(i) * arma::dot(Y(i),r);
        r += S(i) * (alpha(i) - beta);
    }

    p = r;
}

/* lbfgs(f, grad_f, x, max_iter) : Limited memory BFGS algorithm for local minimization
 * --- f  : objective function to minimize
 * --- grad_f : gradient of objective function
 * --- x : initial guess close to a local minimum, root will be stored here
 * --- max_iter : maximum number of iterations after which the solver will stop regardless of convergence. */
void numerics::lbfgs::minimize(const std::function<double(const arma::vec&)>& f,
                               const std::function<arma::vec(const arma::vec&)>& grad_f,
                               arma::vec& x0, int max_iter) {
    if (max_iter <= 0) {
        if (max_iterations <= 0) max_iterations = 100;
    } else max_iterations = max_iter;

    int n = x0.n_elem;
    arma::vec x1, g0, g1, s, y, p;
    double alpha, hdiag;

    g0 = grad_f(x0);
    alpha = wolfe_step(f, grad_f, x0, -g0, wolfe_c1, wolfe_c2);
    x1 = x0 - alpha * g0;
    g1 = grad_f(x1);

    numerics_private_utility::cyc_queue S_historic(n, steps_to_remember), Y_historic(n, steps_to_remember);

    uint iters = 1;
    while (true) {
        if (arma::norm(g0, "inf") < tol) {
            exit_flag = 0;
            num_iter += iters;
            return;
        }

        if (iters >= max_iterations) {
            exit_flag = 1;
            num_iter += iters;
            return;
        }

        s = x1 - x0;
        y = g1 - g0;
        hdiag = arma::dot(s, y) / arma::dot(y, y);

        S_historic.push(s);
        Y_historic.push(y);

        p = -g1;
        
        lbfgs_update(p, S_historic, Y_historic, hdiag);
        if (p.has_nan() || p.has_inf()) {
            p = -grad_f(x0);
            S_historic.clear();
            Y_historic.clear();
        }

        alpha = wolfe_step(f, grad_f, x1, p, wolfe_c1, wolfe_c2);

        if (std::isnan(alpha) || std::isinf(alpha)) {
            exit_flag = 2;
            num_iter += iters;
            return;
        }

        x0 = x1; g0 = g1;
        x1 += alpha * p;
        g1 = grad_f(x1);

        iters++;
    }
}