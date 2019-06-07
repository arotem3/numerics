#include "numerics.hpp"

/* LMLSQR : Levenberg-Marquardt damped least squares algorithm.
 * --- f : function to find least squares solution of.
 * --- x : solution, initialized to a good guess.
 * --- opts : options for controlling solver parameters. */
void numerics::lmlsqr(const vector_func& f, arma::vec& x, lsqr_opts& opts) {
    double tau = opts.damping_param;
    double nu = opts.damping_scale;
    arma::vec delta = 0.01*arma::ones(arma::size(x));
    
    arma::mat J;
    if (opts.jacobian_func == nullptr) {
        approx_jacobian(f,J,x);
        opts.num_FD_approx_made++;
    } else {
        J = opts.jacobian_func->operator()(x);
    }

    arma::mat LSQR_MAT = J.t() * J;
    double lam = tau*arma::max( J.diag() );

    arma::vec F = f(x);

    uint k = 0;
    while (arma::norm(F,"inf") > opts.err) {
        if (k >= opts.max_iter) { // too many iterations needed
            std::cerr << "\nlmlsqr() failed: too many iterations needed for convergence." << std::endl
                      << "Returning current best estimate." << std::endl
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "system norm: " << arma::norm(f(x),"inf") << " > " << opts.err << std::endl << std::endl;
            break;
        }
        
        arma::vec RHS = -J.t() * F;

        double rho;
        do {
            arma::mat LHS;
            if (opts.use_scale_invariance) LHS = LSQR_MAT + lam*arma::diagmat(LSQR_MAT); // J'J + lam*diag(J'J)
            else LHS = LSQR_MAT + lam*arma::eye(arma::size(LSQR_MAT)); // J'J + lam*I
            
            if (opts.use_cgd) cgd(LHS, RHS, delta);
            else delta = arma::solve(LHS, RHS);
            arma::vec F1 = f(x + delta);
            
            rho = (arma::norm(F) - arma::norm(F1));
            rho /= arma::dot(delta, lam*delta + RHS);
            if (rho > 0) {
                x += delta;
                if (opts.jacobian_func == nullptr) { //update jacobian
                    if (arma::norm(F1) < arma::norm(F)) {
                        J += ((F1-F) - J*delta)*delta.t() / arma::as_scalar(delta.t()*delta);
                    } else {
                        approx_jacobian(f,J,x);
                        opts.num_FD_approx_made++;
                    }
                } else {
                    J = opts.jacobian_func->operator()(x);
                }
                LSQR_MAT = J.t() * J;
                F = F1;
                lam *= std::max( 0.33, 1 - std::pow(2*rho-1,3) ); // 1 - (2r-1)^3
                nu = 2;
            } else {
                lam *= nu;
                nu *= 2;
            }
        } while (rho < 0);

        if (arma::norm(delta, "inf") < opts.err) break;
        
        k++;
    }
    opts.num_iters_returned = k;
    opts.final_jacobian = J;
}

/* LMLSQR : Levenberg-Marquardt damped least squares algorithm.
 * --- f : function to find least squares solution of.
 * --- x : solution, initialized to a good guess. */
numerics::lsqr_opts numerics::lmlsqr(const vector_func& f, arma::vec& x) {
    lsqr_opts opts;
    lmlsqr(f,x,opts);
    return opts;
}