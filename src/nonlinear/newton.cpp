#include "numerics.hpp"

/* NEWTON : finds a local root of a multivariate nonlinear system of equations using newton's method.
 * --- f  : system of equations.
 * --- J  : jacobian of system.
 * --- x : initial guess as to where the root, also where the root will be returned to.
 * --- opts : nonlinear options for controlling solver parameters. */
void numerics::newton(const vector_func& f, const vec_mat_func& J, arma::vec& x, nonlin_opts& opts) {    
    arma::vec dx = {1};
    size_t k = 1;
    
    while ( arma::norm(dx, "Inf") > opts.err ) {
        if (k > opts.max_iter) { //newton method takes too long
            std::cerr << "\nnewton() failed: too many iterations needed for convergence." << std::endl
                      << "Returning current best estimate." << std::endl
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "system norm: " << arma::norm(f(x),"inf") << " > " << opts.err << std::endl << std::endl;
                      return;
        }
        dx = -arma::solve(J(x),f(x));
        x += dx;
        k++;
    }

    opts.num_iters_returned = k;
    opts.num_FD_approx_needed = 0;
}

/* NEWTON : finds a local root of a multivariate nonlinear system of equations using newton's method.
 * --- f  : system of equations.
 * --- J  : jacobian of system.
 * --- x : initial guess as to where the root, also where the root will be returned to. */
numerics::nonlin_opts numerics::newton(const vector_func& f, const vec_mat_func& J, arma::vec& x) {
    nonlin_opts opts;
    opts.max_iter = newton_max_iter;

    newton(f,J,x,opts);
    return opts;
}

/* NEWTON : finds a local minimum of nonlinear function using newton's method.
 * --- obj_func : objective function.
 * --- f  : gradient function.
 * --- J  : hessian function.
 * --- x : initial guess for the min, also where the min will be returned to.
 * --- opts : nonlinear options for controlling solver parameters. */
void numerics::newton(const vec_dfunc& obj_func, const vector_func& f, const vec_mat_func& J, arma::vec& x, nonlin_opts& opts) {
    arma::vec s = {1};
    size_t k = 1;
    
    while ( arma::norm(s, "Inf") > opts.err ) {
        if (k > opts.max_iter) { //newton method takes too long
            std::cerr << "\nnewton() failed: too many iterations needed for convergence." << std::endl
                      << "Returning current best estimate." << std::endl
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "system norm: " << arma::norm(f(x),"inf") << " > " << opts.err << std::endl << std::endl;
                      return;
        }
        arma::vec p = -arma::solve(J(x),f(x));
        double alpha = wolfe_step(obj_func,f,x,p,opts.wolfe_c1, opts.wolfe_c2, opts.wolfe_scaling);
        s = alpha*p;
        x += s;
        k++;
    }

    opts.num_iters_returned = k;
    opts.num_FD_approx_needed = 0;
}