#include <numerics.hpp>

/* lbfgs_update(p,S,Y) : update function for limited memory BFGS
 * --- p : negative gradient
 * --- S : s_i = x_i - x_(i-1) 
 * --- Y : y_i = f_i - f_(i-1) */
void numerics::lbfgs::lbfgs_update(arma::vec& p,
                                   numerics_private_utility::cyc_queue& S,
                                   numerics_private_utility::cyc_queue& Y,
                                   const arma::vec& hdiag) {
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

    arma::vec r = q % hdiag;

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
                               arma::vec& x, int max_iter) {
    if (max_iter <= 0) {
        if (max_iterations <= 0) max_iterations = 100;
    } else max_iterations = max_iter;

    int n = x.n_elem;
    arma::vec p, p1, s, y;
    double alpha;
    p = -grad_f(x);
    arma::vec hdiag = 1/jacobian_diag(grad_f, x);
    hdiag(arma::find_nonfinite(hdiag)).zeros();
    alpha = wolfe_step(f,grad_f,x,p,wolfe_c1,wolfe_c2,wolfe_scale);
    s = alpha*p;
    x += s;
    p1 = -grad_f(x);
    y = p1 - p;
    p = -p1;

    numerics_private_utility::cyc_queue S_historic(n, steps_to_remember), Y_historic(n, steps_to_remember);
    S_historic.push(s);
    Y_historic.push(y);

    uint k = 1;
    do {
        if (k >= max_iterations) {
            exit_flag = 1;
            num_iter += k;
            return;
        }
        
        lbfgs_update(p, S_historic, Y_historic, hdiag);
        if (p.has_nan() || p.has_inf()) {
            p = -grad_f(x);
            S_historic.clear();
            Y_historic.clear();
            hdiag = 1/jacobian_diag(grad_f, x);
            hdiag(arma::find_nonfinite(hdiag)).zeros();
        }
        alpha = wolfe_step(f,grad_f,x,p,wolfe_c1,wolfe_c2,wolfe_scale);
        s = alpha*p;

        if (s.has_nan() || s.has_inf()) {
            num_iter += k;
            exit_flag = 2;
            return;
        }

        x += s;
        p1 = grad_f(x);
        y = -p1 - p;
        p = -p1;

        S_historic.push(s);
        Y_historic.push(y);

        k++;
    } while (arma::norm(s,"inf") > tol);
    num_iter += k;
    exit_flag = 0;
}