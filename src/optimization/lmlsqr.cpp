#include <numerics.hpp>

/* lmlsqr(f, x, max_iter) : Levenberg-Marquardt damped least squares algorithm.
 * --- f : function to find least squares solution of.
 * --- x : solution, initialized to a good guess.
 * --- max_iter : maximum allowed iterations. */
void numerics::lmlsqr::fsolve(const std::function<arma::vec(const arma::vec& x)>& f,
                             arma::vec& x,
                             int max_iter) {
    if (max_iter <= 0) {
        if (max_iterations <= 0) max_iterations = 100;
    } else max_iterations = max_iter;
    
    double tau = damping_param, nu = damping_scale;
    arma::vec F1, delta = 0.01*arma::ones(arma::size(x));

    arma::mat J = approx_jacobian(f,x);
    arma::vec F = f(x);

    arma::vec D; arma::mat V;
    if (!use_cgd) arma::eig_sym(D, V, J.t() * J);
    double lam = tau*arma::norm(J.diag(),"inf");

    uint k = 0;
    do {
        if (k >= max_iterations) {
            exit_flag = 1;
            num_iter += k;
            return;
        }

        arma::vec RHS = -(J.t() * F);
        double rho;
        do {
            arma::mat LHS;
            if (use_cgd) {
                LHS = J.t() * J;
                LHS.diag() += lam;
                numerics::cgd(LHS, RHS, delta);
            } else {
                delta = V * arma::diagmat(1/(D+lam)) * V.t() * RHS;
            }
            
            if (delta.has_nan() || delta.has_inf()) {
                exit_flag = 2;
                num_iter += k;
                return;
            }
            F1 = f(x + delta);

            rho = (arma::norm(F) - arma::norm(F1));
            rho /= arma::dot(delta, lam*delta + RHS);
            if (rho > 0) { // new point is acceptible
                x += delta;
                J += ((F1-F) - J*delta)*delta.t() / arma::dot(delta, delta);
                if (J.has_nan() || J.has_inf()) J = approx_jacobian(f,x);
                if (!use_cgd) arma::eig_sym(D, V, J.t() * J);
                F = F1;
                lam *= std::max( 0.33, 1 - std::pow(2*rho-1,3) ); // 1 - (2r-1)^3
                nu = 2;
            } else { // need to shrink trust region
                lam *= nu;
                nu *= 2;
            }
            k++;
        } while(rho < 0);
    } while (arma::norm(delta,"inf") > tol);

    num_iter += k;
    exit_flag = 0;

    return;
}

/* lmlsqr(f, x, max_iter) : Levenberg-Marquardt damped least squares algorithm.
 * --- f : f(x) == 0 function to find least squares solution of.
 * --- jacobian : J(x) jacobian function of f(x).
 * --- x : solution, initialized to a good guess.
 * --- max_iter : maximum allowed iterations. */
void numerics::lmlsqr::fsolve(const std::function<arma::vec(const arma::vec& x)>& f,
                             const std::function<arma::mat(const arma::vec& x)>& jacobian,
                             arma::vec& x,
                             int max_iter) {
    if (max_iter <= 0) {
        if (max_iterations <= 0) max_iterations = 100;
    } else max_iterations = max_iter;
    
    double tau = damping_param, nu = damping_scale;
    arma::vec F, F1, delta = 0.01*arma::ones(arma::size(x));
    arma::mat J;

    J = jacobian(x);
    F = f(x);

    double lam = tau*arma::norm(J.diag(),"inf");
    arma::mat V; arma::vec D;
    if (!use_cgd) {
        arma::eig_sym(D, V, J.t()*J);
    }

    uint k = 0;
    do {
        if (k >= max_iterations) {
            exit_flag = 1;
            num_iter += k;
            return;
        }

        arma::vec RHS = -(J.t() * F);
        double rho;
        do {
            arma::mat LHS;
            if (use_cgd) {
                LHS = J.t() * J;
                LHS.diag() += lam;
                numerics::cgd(LHS, RHS, delta);
            } else {
                delta = V * arma::diagmat(1/(D+lam)) * V.t() * RHS;
            }
            
            if (delta.has_nan() || delta.has_inf()) {
                exit_flag = 2;
                num_iter += k;
                return;
            }
            F1 = f(x + delta);

            rho = (arma::norm(F) - arma::norm(F1));
            rho /= arma::dot(delta, lam*delta + RHS);
            if (rho > 0) {
                x += delta;
                J = jacobian(x);
                if (!use_cgd) arma::eig_sym(D, V, J.t()*J);
                F = F1;
                lam *= std::max( 0.33, 1 - std::pow(2*rho-1,3) ); // 1 - (2r-1)^3
                nu = 2;
            } else {
                lam *= nu;
                nu *= 2;
            }
            k++;
        } while(rho < 0);
    } while (arma::norm(delta,"inf") > tol);

    num_iter += k;
    exit_flag = 0;

    return;
}