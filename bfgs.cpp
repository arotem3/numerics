#include "numerics.hpp"

//--- rough approximator step size for quasi-newton methods based on strong wolfe conditions ---//
double numerics::wolfe_step(const vec_dfunc& obj_func, const vector_func& f, const arma::vec& x, const arma::vec& p, double c1, double c2, double b) {
    double alpha = 1;
    int k = 0;
    while (true) {
        if (k > 100) break;
        double pfa = arma::dot(p, f(x + alpha*x));
        bool cond1 = obj_func(x + alpha*p) <= obj_func(x) + c1*alpha*pfa;
        bool cond2 = std::abs(arma::dot(p, f(x + alpha*p))) <= c2*std::abs(arma::dot(p, f(x)));
        if (cond1 && cond2) break;
        else if (obj_func(x + b*alpha*p) < obj_func(x + alpha*p)) alpha *= b;
        else alpha /= b;
        k++;
    }
    return alpha;
}


//--- Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm for local optimization ---//
//----- f  : vec = f(x) system of equations --------------------------------------//
//----- x : initial guess close to a local minimum, root will be stored here -----//
//----- opts : nonlinear options for controlling solver parameters ---------------//
void numerics::bfgs(const vec_dfunc& obj_func, const vector_func& f, arma::vec& x, nonlin_opts& opts) {
    opts.num_FD_approx_needed = 0;
    
    //---(0) determine size of vector
    int n = x.n_elem;
    
    //---(1.a) initialize inverse Hessian B^-1 = H
    arma::mat H(n,n,arma::fill::eye);
    if (opts.init_jacobian_inv != nullptr) H = *opts.init_jacobian_inv;
    else if (opts.init_jacobian != nullptr) {
        H = *opts.init_jacobian;
        H = arma::pinv(H);
    } else if (opts.use_FD_jacobian) {
        approx_jacobian(f,H,x);
        H = arma::pinv(H);
        opts.num_FD_approx_needed++;
    } // else keep it as identity

    //---(1.b) compute search direction, p.
    arma::vec p = -H*f(x);
    double alpha = wolfe_step(obj_func,f,x,p,opts.wolfe_c1, opts.wolfe_c2, opts.wolfe_scaling);
    arma::vec s = alpha*p;
    //---(1.c) update x
    x += s;
    //---(1.d) define y = f(x1) - f(x)
    arma::vec y = f(x) - f(x-s);
    //---(1.e) define next B:
    H = H + (1 + arma::dot(y, H*y)/arma::dot(s,y))*(s*s.t())/arma::dot(s,y) - (s*y.t()*H + H*y*s.t())/arma::dot(y,s);
    //---(2) iterate this process.
    size_t k = 1;
    while ( arma::norm(s,"inf") > opts.err ) {
        //---check if bfgs takes too long
        if (k >= opts.max_iter) {
            std::cerr << "\nbfgs() failed: too many iterations needed for convergence." << std::endl
                 << "returning current best estimate." << std::endl
                 << "!!!---not necessarily a good estimate---!!!" << std::endl
                 << "inf norm of f(x) = " << arma::norm(f(x), "inf") << " > 0" << std::endl << std::endl;
            return;
        }

        p = -H*f(x);
        alpha = wolfe_step(obj_func,f,x,p,opts.wolfe_c1, opts.wolfe_c2, opts.wolfe_scaling);
        s = alpha*p;

        x += s;
        y = f(x) - f(x-s);

        // update H based on conditions
        if ( opts.use_FD_jacobian && (arma::norm(f(x)) > arma::norm(f(x-s))) ) { // H has become inaccurate, FD approx needed
            approx_jacobian(f,H,x);
            H = arma::pinv(H);
            opts.num_FD_approx_needed++;
        } else { // BFGS update is suitable.
            double sy = arma::dot(s,y);
            H = H + (1 + arma::dot(y, H*y)/sy)*(s*s.t())/sy - (s*y.t()*H + H*y*s.t())/sy;   
        }
        k++;
    }
    opts.num_iters_returned = k;
    opts.final_jacobian = H;
    return;
}

numerics::nonlin_opts numerics::bfgs(const vec_dfunc& obj_func, const vector_func& f, arma::vec& x) {
    nonlin_opts opts;
    opts.max_iter = bfgs_max_iter;

    bfgs(obj_func,f,x,opts);
    return opts;
}


//--- update function for limited memory BFGS ---//
arma::vec lbfgs_update(arma::vec& p, numerics::cyc_queue& S, numerics::cyc_queue& Y, arma::vec& hdiags) {
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

    arma::vec r = hdiags % q;

    for (int i(0); i < k; ++i) {
        double beta = ro(i) * arma::dot(Y(i),r);
        r += S(i) * (alpha(i) - beta);
    }

    return r;
}

//--- Limited memory BFGS algorithm for local optimization -----------------------//
//----- f  : vec = f(x) system of equations --------------------------------------//
//----- x : initial guess close to a local minimum, root will be stored here -----//
//----- opts : nonlinear options for controlling solver parameters ---------------//
void numerics::lbfgs(const vec_dfunc& obj_func, const vector_func& f, arma::vec& x, lbfgs_opts& opts) {
    int n = x.n_elem;
    
    arma::vec hdiags;
    if (opts.init_hess_diag_inv.is_empty()) hdiags = arma::ones(n);
    else hdiags = opts.init_hess_diag_inv;

    arma::vec p = -f(x);
    double alpha = wolfe_step(obj_func,f,x,p,opts.wolfe_c1, opts.wolfe_c2, opts.wolfe_scaling);
    arma::vec s = alpha*p;
    x += s;
    arma::vec y = f(x) - f(x-s);

    cyc_queue S_history(n, opts.num_iters_to_remember);
    cyc_queue Y_history(n, opts.num_iters_to_remember);
    S_history.push(s);
    Y_history.push(y);

    size_t k = 0;
    while (arma::norm(s,"inf") > opts.err) {
        if (k > opts.max_iter) {
            std::cerr << "\nlbfgs() failed: too many iterations needed for convergence." << std::endl
                 << "returning current best estimate." << std::endl
                 << "!!!---not necessarily a good estimate---!!!" << std::endl
                 << "inf norm of f(x) = " << arma::norm(f(x), "inf") << " > 0" << std::endl << std::endl;
            return;
        }
        p = -f(x);
        p = lbfgs_update(p, S_history, Y_history, hdiags);
        alpha = wolfe_step(obj_func,f,x,p,opts.wolfe_c1, opts.wolfe_c2, opts.wolfe_scaling);
        s = alpha*p;
        x += s;
        y = f(x) - f(x-s);

        S_history.push(s);
        Y_history.push(y);

        k++;
    }
    opts.num_iters_returned = k;
}

numerics::lbfgs_opts numerics::lbfgs(const vec_dfunc& obj_func, const vector_func& f, arma::vec& x) {
    lbfgs_opts opts;
    lbfgs(obj_func, f, x, opts);
    return opts;
}