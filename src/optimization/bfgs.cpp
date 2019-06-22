#include <numerics.hpp>

/* minimize(f, grad_f, x, max_iter) : Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm for local optimization
 * --- f : f(x) objective function
 * --- grad_f  : gradient of f(x).
 * --- x : initial guess close to a local minimum, root will be stored here.
 * --- max_iter : maximum number of iteration before premature stop */
void numerics::bfgs::minimize(std::function<double(const arma::vec&)> f,
                              std::function<arma::vec(const arma::vec&)> grad_f,
                              arma::vec& x, int max_iter) {
    if (max_iter <= 0) {
        if (max_iterations <= 0) max_iterations = 100;
    } else max_iterations = max_iter;

    int n = x.n_elem;
    arma::mat H;
    arma::vec g, g1;
    g = grad_f(x);
    if (use_finite_difference_hessian) {
        H = arma::symmatu( approx_jacobian(grad_f,x) );
        bool chol_success = arma::inv_sympd(H,H);
        if (!chol_success) H = arma::pinv(H);
    } else H = arma::eye(n,n);

    arma::vec p;
    double alpha;
    arma::vec s, y;
    uint k = 0;
    do {
        if (k > max_iterations) {
            exit_flag = 1;
            num_iter += k;
            return;
        }
        p = -(H*g);
        if (p.has_nan() || p.has_inf()) {
            if (use_finite_difference_hessian) {
                H = arma::symmatu( approx_jacobian(grad_f,x) );
                bool chol_success = arma::inv_sympd(H,H);
                if (!chol_success) H = arma::pinv(H);
            } else H = arma::eye(n,n);
            p = -(H*g);
        }
        alpha = numerics::wolfe_step(f,grad_f,x,p,wolfe_c1,wolfe_c2,wolfe_scale);
        s = alpha*p;

        if (s.has_nan() || s.has_inf()) {
            num_iter += k;
            exit_flag = 2;
            return;
        }

        x += s;
        g1 = grad_f(x);
        y = g1 - g;

        double sdoty = arma::dot(s,y);
        arma::vec Hdoty = H*y;
        H += (1 + arma::dot(y,Hdoty) / sdoty) * (s*s.t())/sdoty - (s*Hdoty.t() + Hdoty*s.t())/sdoty;
        g = g1;
        k++;
    } while (arma::norm(p,"inf") > tol);
    num_iter += k;
    exit_flag = 0;
}

/* minimize(f, grad_f, hessian, x, max_iter) : Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm for local optimization
 * --- f : f(x) objective function
 * --- grad_f  : gradient of f(x).
 * --- hessian : hessian matrix of f(x).
 * --- x : initial guess close to a local minimum, root will be stored here.
 * --- max_iter : maximum number of iteration before premature stop */
void numerics::bfgs::minimize(std::function<double(const arma::vec&)> f,
                              std::function<arma::vec(const arma::vec&)> grad_f,
                              std::function<arma::mat(const arma::vec&)> hessian,
                              arma::vec& x, int max_iter) {
    if (max_iter <= 0) {
        if (max_iterations <= 0) max_iterations = 100;
    } else max_iterations = max_iter;

    int n = x.n_elem;
    arma::mat H;
    arma::vec g, g1;
    g = grad_f(x);
    H = hessian(x);
    bool chol_success = arma::inv_sympd(H,H);
    if (!chol_success) H = arma::pinv(H);

    arma::vec p;
    double alpha;
    arma::vec s, y;
    uint k = 0;
    do {
        if (k > max_iterations) {
            exit_flag = 1;
            num_iter += k;
            return;
        }
        p = -(H*g);
        if (p.has_nan() || p.has_inf()) {
            H = hessian(x);
            bool chol_success = arma::inv_sympd(H,H);
            if (!chol_success) H = arma::pinv(H);
            p = -(H*g);
        }
        alpha = numerics::wolfe_step(f,grad_f,x,p,wolfe_c1,wolfe_c2,wolfe_scale);
        s = alpha*p;

        if (s.has_nan() || s.has_inf()) {
            num_iter += k;
            exit_flag = 2;
            return;
        }

        x += s;
        g1 = grad_f(x);
        y = g1 - g;

        double sdoty = arma::dot(s,y);
        arma::vec Hdoty = H*y;
        H += (1 + arma::dot(y,Hdoty) / sdoty) * (s*s.t())/sdoty - (s*Hdoty.t() + Hdoty*s.t())/sdoty;
        g = g1;
        k++;
    } while (arma::norm(p,"inf") > tol);
    num_iter += k;
    exit_flag = 0;
}