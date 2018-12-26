#include "numerics.hpp"

//--- Levenberg-Marquardt damped least squares algorithm ---//
//----- f : function to find least squares solution of -----//
//----- x : solution, initialized to a good guess ----------//
//----- opts : options for controlling solver parameters ---//
void numerics::lmlsqr(const vector_func& f, arma::vec& x, lsqr_opts& opts) {
    double lam = opts.damping_param;
    double nu = opts.damping_scale;
    arma::vec delta = arma::ones(arma::size(x));
    
    arma::mat J;
    if (opts.jacobian_func == nullptr) {
        approx_jacobian(f,J,x);
        opts.num_FD_approx_made++;
    } else {
        J = opts.jacobian_func->operator()(x);
    }

    size_t k = 0;
    while (arma::norm(delta,"inf") > opts.err) {
        if (k > opts.max_iter) { // too many iterations needed
            std::cerr << "\nnewton() failed: too many iterations needed for convergence." << std::endl
                      << "Returning current best estimate." << std::endl
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "system norm: " << arma::norm(f(x),"inf") << " > " << opts.err << std::endl << std::endl;
            return;
        }
        
        arma::vec F = f(x);
        arma::vec RHS = -J.t() * F;
        arma::mat LSQR_MAT = J.t() * J;
        arma::mat LHS;
        if (opts.use_scale_invariance) LHS = LSQR_MAT + lam*arma::diagmat(LSQR_MAT); // J'J + lam*diag(J'J)
        else LHS = LSQR_MAT + lam*arma::eye(arma::size(LSQR_MAT)); // J'J + lamI

        delta = arma::solve(LHS, RHS);

        arma::vec F1 = f(x + delta);
        if (arma::norm(F1, "inf") < arma::norm(F,"inf")) { // delta is sufficient so we can undate
            x += delta;
            if (opts.jacobian_func == nullptr) { //update jacobian
                approx_jacobian(f,J,x);
                opts.num_FD_approx_made++;
            } else {
                J = opts.jacobian_func->operator()(x);
            }
            lam /= nu;
        } else { // delta is insufficient
            lam *= nu;
        }
        k++;
    }
    opts.num_iters_returned = k;
    opts.final_jacobian = J;
}

numerics::lsqr_opts numerics::lmlsqr(const vector_func& f, arma::vec& x) {
    lsqr_opts opts;
    lmlsqr(f,x,opts);
    return opts;
}