#include "numerics.hpp"

//--- trust region based nonlinear conjugate gradient method ---//
//----- f : gradient function ----------------------------------//
//----- x : guess, and solution --------------------------------//
void numerics::nlcgd(const vector_func& f, arma::vec& x, nonlin_opts& opts) {
    int n = x.n_elem;
    
    arma::vec p = -f(x);
    arma::vec s = p;
    
    arma::mat I = arma::eye(n,n);
    arma::mat B;
    if (opts.init_jacobian == nullptr) { // no initial jacobian
        if (opts.jacobian_func != nullptr) { // jacobian function provided
            B = opts.jacobian_func->operator()(x);
        } else if (opts.use_FD_jacobian) { // FD jacobian
            approx_jacobian(f,B,x);
        } else { // identity
            B = arma::eye(n,n);
        }
    } else { // initial jacobian
        B = *opts.init_jacobian;
    }

    size_t k = 0;
    while (arma::norm(s,"inf") > opts.err) {
        if (k >= opts.max_iter) {
            std::cerr << "\nnlcgd() failed: too many iterations needed for convergence." << std::endl
                 << "returning current best estimate." << std::endl
                 << "!!!---not necessarily a good estimate---!!!" << std::endl
                 << "||f(x)|| = " << arma::norm(f(x), "inf") << " > 0" << std::endl << std::endl;
            break;
        }
        arma::vec fx = f(x);
        double alpha = arma::dot(fx,fx) / arma::dot(p, B*p);
        double sts = arma::dot(s,s);
        arma::vec ds = alpha*p - s;

        s = alpha*p;
        x += s;
        k++;

        arma::vec fx1 = -f(x);
        if (mod(k,n) == 0) {
            p = fx1;
            if (opts.jacobian_func == nullptr) { // BFGS update
                arma::vec y = -(fx1 + fx);
                double sy = arma::dot(s,y);
                arma::vec BS = B*s;
                B += y*y.t()/sy - BS*BS.t()/arma::dot(s,BS);
            } else { // Hessian provided
                B = opts.jacobian_func->operator()(x);
            }
        } else {
            double beta = std::max(arma::dot(s,ds)/sts, 0.0);
            p = fx1 + beta*p;
        }

        // if (opts.jacobian_func == nullptr) { // BFGS update
        //     arma::vec y = -(fx1 + fx);
        //     double sy = arma::dot(s,y);
        //     arma::vec BS = B*s;
        //     B += y*y.t()/sy - BS*BS.t()/arma::dot(s,BS);
        // } else { // Hessian provided
        //     B = opts.jacobian_func->operator()(x);
        // }
        
    }

    opts.num_iters_returned = k;
}

numerics::nonlin_opts numerics::nlcgd(const vector_func& f, arma::vec& x) {
    nonlin_opts opts;
    nlcgd(f,x,opts);
    return opts;
}