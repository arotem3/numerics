#include "numerics.hpp"

//--- finds a local root of a multivariate nonlinear system of equations using newton's method ---//
//----- f  : system of equations -----------------------------------------------------------------//
//----- J  : jacobian of system ------------------------------------------------------------------//
//----- x : initial guess as to where the root, also where the root will be returned to ----------//
//----- opts : nonlinear options for controlling solver parameters -------------------------------//
void numerics::newton(std::function<arma::vec(const arma::vec&)> f, std::function<arma::mat(const arma::vec&)> J, arma::vec& x, nonlin_opts& opts) {    
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

numerics::nonlin_opts numerics::newton(std::function<arma::vec(const arma::vec&)> f, std::function<arma::mat(const arma::vec&)> J, arma::vec& x) {
    nonlin_opts opts;
    opts.max_iter = newton_max_iter;

    newton(f,J,x,opts);
    return opts;
}

//--- Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm for local root of nonlinear system of equations ---//
//----- f  : vec = f(x) system of equations ---------------------------------------------------------------//
//----- x : initial guess close to a local minimum, root will be stored here ------------------------------//
//----- opts : nonlinear options for controlling solver parameters ----------------------------------------//
void numerics::bfgs(std::function<arma::vec(const arma::vec&)> f, arma::vec& x, nonlin_opts& opts) {
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

    //---(1.b) compute change in x (s).
    arma::vec s = -H*f(x);
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
        if (k > opts.max_iter) {
            std::cerr << "\nbfgs() failed: too many iterations needed for convergence." << std::endl
                 << "returning current best estimate." << std::endl
                 << "!!!---not necessarily a good estimate---!!!" << std::endl
                 << "inf norm of f(x) = " << arma::norm(f(x), "inf") << " > 0" << std::endl << std::endl;
            return;
        }

        s = -H*f(x);

        x += s;
        y = f(x) - f(x-s);

        // update H based on conditions
        if ( opts.use_FD_jacobian && (arma::norm(f(x)) > arma::norm(f(x-s))) ) { // H has become inaccurate, FD approx needed
            approx_jacobian(f,H,x);
            H = arma::pinv(H);
            opts.num_FD_approx_needed++;
        } else { // BFGS update is suitable.
            H = H + (1 + arma::dot(y, H*y)/arma::dot(s,y))*(s*s.t())/arma::dot(s,y) - (s*y.t()*H + H*y*s.t())/arma::dot(y,s);   
        }
        k++;
    }
    opts.num_iters_returned = k;
    opts.final_jacobian = H;
    return;
}

numerics::nonlin_opts numerics::bfgs(std::function<arma::vec(const arma::vec&)> f, arma::vec& x) {
    nonlin_opts opts;
    opts.max_iter = bfgs_max_iter;

    bfgs(f,x,opts);
    return opts;
}

//--- Broyden's method for local root finding of nonlinear system of equations ---//
void numerics::broyd(std::function<arma::vec(const arma::vec&)> f, arma::vec& x, nonlin_opts& opts) {
    opts.num_FD_approx_needed = 0;

    arma::mat Jinv;
    //--- (1) initialize approximation of inverse Jacobian
    if (opts.init_jacobian_inv != nullptr) Jinv = *opts.init_jacobian_inv;
    else if (opts.init_jacobian != nullptr) {
        Jinv = *opts.init_jacobian;
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
            approx_jacobian(f,Jinv,x);
            Jinv = arma::pinv(Jinv);
            opts.num_FD_approx_needed++;
        } else { // broyden update is suitable
            y = f(x) - F;
            Jinv += (dx - Jinv*y)*dx.t()*Jinv/arma::dot(dx,Jinv*y);
        }
        k++;
    }
    opts.num_iters_returned = k;
    opts.final_jacobian = Jinv;
}

numerics::nonlin_opts numerics::broyd(std::function<arma::vec(const arma::vec&)> f, arma::vec& x) {
    nonlin_opts opts;
    opts.err = root_err;
    opts.max_iter = broyd_max_iter;

    broyd(f,x,opts);
    return opts;
}