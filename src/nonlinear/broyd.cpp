#include "numerics.hpp"

/* BROYD : Broyden's method for local root finding of nonlinear system of equations
 * --- f : f(x) = 0 function for finding roots of
 * --- x : guess for root, also where root will be stored
 * --- opts : nonlinear options, such as tolerance and max allowed iterations */
void numerics::broyd(const vector_func& f, arma::vec& x, nonlin_opts& opts) {
    opts.num_FD_approx_needed = 0;

    arma::mat Jinv;
    //--- (1) initialize approximation of inverse Jacobian
    if (opts.init_jacobian_inv != nullptr) Jinv = *opts.init_jacobian_inv;
    else if (opts.init_jacobian != nullptr) {
        Jinv = *opts.init_jacobian;
        Jinv = arma::pinv(Jinv);
    } else if (opts.jacobian_func != nullptr) {
        Jinv = opts.jacobian_func->operator()(x);
        Jinv = arma::pinv(Jinv);
    } else { // FD approximation always used to initialize.
        approx_jacobian(f,Jinv,x,1e-4);
        Jinv = arma::pinv(Jinv);
        opts.num_FD_approx_needed++;
    }

    //--- (2) compute f(x) and change in x (dx)
    arma::vec F = f(x);

    arma::vec dx = -Jinv*F;
    x += dx;

    //--- (3) update inverse approximation
    arma::vec y = f(x) - F; // change in f(x), needed for inverse update
    Jinv += (dx - Jinv*y)*dx.t()*Jinv/arma::dot(dx,Jinv*y);

    size_t k = 1;
    while (arma::norm(dx,"inf") > opts.err) {
        if (k > opts.max_iter) { // stop if takes too long
            std::cerr << "\nbroyd() failed: too many iterations needed for convergence." << std::endl
                 << "returning current best estimate." << std::endl
                 << "!!!---not necessarily a good estimate---!!!" << std::endl
                 << "inf norm of f(x) = " << arma::norm(f(x), "inf") << " > 0" << std::endl << std::endl;
            return;
        }
        F = f(x);
        dx = -Jinv * F;
        x += dx;
        // update inverse jacobian based on current condition of the approximation
        if ( opts.use_FD_jacobian && (arma::norm(f(x)) > arma::norm(F)) ) { // Jinv has become innaccurate, FD approx needed
            if (opts.jacobian_func != nullptr) {
                Jinv = opts.jacobian_func->operator()(x);
                Jinv = arma::pinv(Jinv);
            } else {
                approx_jacobian(f,Jinv,x);
                Jinv = arma::pinv(Jinv);
                opts.num_FD_approx_needed++;
            }
        } else { // broyden update is suitable
            y = f(x) - F;
            Jinv += (dx - Jinv*y)*dx.t()*Jinv/arma::dot(dx,Jinv*y);
        }
        k++;
    }
    opts.num_iters_returned = k;
    opts.final_jacobian = Jinv;
}

/* BROYD : Broyden's method for local root finding of nonlinear system of equations
 * --- f : f(x) = 0 function for finding roots of
 * --- x : guess for root, also where root will be stored */
numerics::nonlin_opts numerics::broyd(const vector_func& f, arma::vec& x) {
    nonlin_opts opts;
    opts.err = root_err;
    opts.max_iter = broyd_max_iter;

    broyd(f,x,opts);
    return opts;
}