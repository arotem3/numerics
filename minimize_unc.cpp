#include "numerics.hpp"

//--- unconstrained minimization of multivariate function ---//
//----- f : objective function ------------------------------//
//----- x : initial guess, also where minimum is stored -----//
//----- opts: optimization options --------------------------//
double numerics::minimize_unc(const vec_dfunc& f, arma::vec& x, optim_opts& opts) {
    if ((opts.use_FD_gradient == false) && (opts.gradient_func == nullptr) && (opts.indexed_gradient_func == nullptr)) { // error
        std::cerr << "minimize_unc() error: conflicting optimization options." << std::endl
                  << "you must either provide a gradient function, or enable finite difference evaluations" << std::endl;
        return NAN;
    }

    if ((opts.use_FD_gradient == true) && (opts.gradient_func == nullptr) && (opts.indexed_gradient_func == nullptr)) { // warning about finite differences
        std::cerr << "minimize_unc() warning: using finite differences for optimization can be extremely inefficient, but proceeding anyways." << std::endl
                  << "consider providing a gradient function or using genOptim() instead if minimize_unc() fails." << std::endl
                  << "as a general guideline for unconstrained optimization:" << std::endl
                  << "1) in most cases Newton's method requires the least number gradient evaluations," << std::endl
                  << "\tbut the method requires a Hessian function which may be inefficient to evaluate" << std::endl
                  << "2) BFGS is very powerful as well, but can be storage inefficient when optimizing in many variables," << std::endl
                  << "\tthe method also requires a gradient function and is most efficient if the Hessian at the intial point is provided," << std::endl
                  << "\talternatively enabling FD hessian evaluations can be efficient for well conditioned objective functions." << std::endl
                  << "3) L-BFGS (or limited memory BFGS) has the benefits of BFGS but is more memory efficient." << std::endl
                  << "\tif memory is a problem then L-BFGS should be used over BFGS, though BFGS typically converges slightly faster." << std::endl
                  << "4) The nonlinear conjugate gradient method is optimal when optimizing in few variables, and when hessian computations are expensive" << std::endl
                  << "5) Levenberg-Marquardt behaves similarly to BFGS and is the prefered method for nonlinear least squares problems," << std::endl
                  << "\tif you are solving a least squares problem, consider instead using lmlsqr() and providing the system g_i(b) = y_i - f(x_i;b)" << std::endl
                  << "6) Momentum gradient descent is used when Hessian based computations are too expensive," << std::endl
                  << "\tthis method typically requires many more iterations than Hessian based methods" << std::endl
                  << "7) Stochastic gradient descent is used when even gradient based computations are too expensive." << std::endl
                  << "\tchoosing a batch size and max number of iterations are all important choices for the success of this algorithm" << std::endl
                  << "\tFor generic root finding use Broyden's method provided by broyd()." << std::endl
                  << std::endl << "\t\t\tGood Luck!" << std::endl;
    }

    if (opts.solver == MGD) { // using momentum gradient descent
        gd_opts options;
        options.err = opts.tolerance;
        options.max_iter = opts.max_iter;
        options.damping_param = opts.damping_param;
        if (opts.gradient_func == nullptr) { // no gradient provided
            std::cerr << "minimize_unc() warning: using FD gradients may be very slow!" << std::endl;
            options.max_iter = no_grad_max_iter;
            auto df = [f,opts](const arma::vec& u) -> arma::vec {return grad(f,u,opts.tolerance);};
            mgd(df,x,options);
        } else { // gradient provided -- standard momentum
            mgd(*opts.gradient_func, x, options);
        }
        opts.num_iters_returned = options.num_iters_returned;
    } else if (opts.solver == BFGS) { // using BFGS
        nonlin_opts options;
        options.max_iter = opts.max_iter;
        options.err = opts.tolerance;
        options.init_jacobian = opts.init_hessian;
        options.init_jacobian_inv = opts.init_hessian_inv;
        options.jacobian_func = opts.hessian_func;
        options.use_FD_jacobian = opts.use_FD_hessian;
        options.wolfe_c1 = opts.wolfe_c1;
        options.wolfe_c2 = opts.wolfe_c2;
        options.wolfe_scaling = opts.wolfe_scaling;
        if (opts.gradient_func == nullptr) { // no gradient function
            std::cerr << "minimize_unc() warning: using FD gradients may be very slow!" << std::endl;
            options.max_iter = no_grad_max_iter;
            auto df = [f,opts](const arma::vec& u) -> arma::vec {return grad(f,u,opts.tolerance);};
            bfgs(f, df, x, options);
        } else {
            bfgs(f, *opts.gradient_func, x, options);
        }
        opts.num_iters_returned = options.num_iters_returned;
    } else if (opts.solver == LBFGS) { // using limited memory BFGS
        lbfgs_opts options;
        options.max_iter = opts.max_iter;
        options.err = opts.tolerance;
        options.num_iters_to_remember = opts.num_iters_to_remember;
        options.wolfe_c1 = opts.wolfe_c1;
        options.wolfe_c2 = opts.wolfe_c2;
        options.wolfe_scaling = opts.wolfe_scaling;
        if (opts.init_hessian_inv != nullptr) {
            if (opts.init_hessian_inv->is_empty() || opts.init_hessian_inv->is_colvec()) {
                options.init_hess_diag_inv = *opts.init_hessian_inv;
            } else {
                options.init_hess_diag_inv = opts.init_hessian_inv->diag();
            }
        }
        if (opts.gradient_func == nullptr) { // no gradient function
            std::cerr << "minimize_unc() warning: using FD gradients may be very slow!" << std::endl;
            options.max_iter = no_grad_max_iter;
            auto df = [f,opts](const arma::vec& u) -> arma::vec {return grad(f,u,opts.tolerance);};
            lbfgs(f, df, x, options);
        } else {
            lbfgs(f, *opts.gradient_func, x, options);
        }
        opts.num_iters_returned = options.num_iters_returned;
    } else if (opts.solver == LMLSQR) { // using Levenberg-Marquardt
        lsqr_opts options;
        options.err = opts.tolerance;
        options.max_iter = opts.max_iter;
        options.damping_param = opts.damping_param;
        options.damping_scale = opts.damping_scale;
        options.jacobian_func = opts.hessian_func;
        options.use_scale_invariance = opts.use_scale_invariance;
        if (opts.gradient_func == nullptr) {
            std::cerr << "minimize_unc() warning: using FD gradients may be very slow!" << std::endl;
            options.max_iter = no_grad_max_iter;
            auto df = [f,opts](const arma::vec& u) -> arma::vec {return grad(f,u,opts.tolerance);};
            lmlsqr(df,x,options);
        } else {
            lmlsqr(*opts.gradient_func, x, options);
        }
        opts.num_iters_returned = options.num_iters_returned;
    } else if (opts.solver == BROYD) { // using Broyden's method
        std::cerr << "minimize_unc() warning: Broyden's method may not as optimal as BFGS, consider changing to BFGS if Broyden is unsuccessful." << std::endl;
        nonlin_opts options;
        options.max_iter = opts.max_iter;
        options.err = opts.tolerance;
        options.init_jacobian = opts.init_hessian;
        options.init_jacobian_inv = opts.init_hessian_inv;
        options.jacobian_func = opts.hessian_func;
        options.use_FD_jacobian = opts.use_FD_hessian;
        if (opts.gradient_func == nullptr) { // no gradient function
            std::cerr << "minimize_unc() warning: using FD gradients may be very slow!" << std::endl;
            options.max_iter = no_grad_max_iter;
            auto df = [f,opts](const arma::vec& u) -> arma::vec {return grad(f,u,opts.tolerance);};
            broyd(df,x,options);
        } else {
            broyd(*opts.gradient_func, x, options);
        }
        opts.num_iters_returned = options.num_iters_returned;
    } else if (opts.solver == NEWTON) { // using Newton's method
        nonlin_opts options;
        options.max_iter = opts.max_iter;
        options.err = opts.tolerance;
        options.wolfe_c1 = opts.wolfe_c1;
        options.wolfe_c2 = opts.wolfe_c2;
        options.wolfe_scaling = opts.wolfe_scaling;
        if (opts.hessian_func == nullptr) { // switching to BFGS
            std::cerr << "minimize_unc() warning: Newton's method requires a Hessian function and one was not provided. Using BFGS instead." << std::endl;
            opts.solver = BFGS;
            options.init_jacobian = opts.init_hessian;
            options.init_jacobian_inv = opts.init_hessian_inv;
            if (opts.gradient_func == nullptr) { // no gradient function
                std::cerr << "minimize_unc() warning: using FD gradients may be very slow!" << std::endl;
                options.max_iter = no_grad_max_iter;
                auto df = [f,opts](const arma::vec& u) -> arma::vec {return grad(f,u,opts.tolerance);};
                bfgs(f, df, x, options);
            } else {
                bfgs(f, *opts.gradient_func, x, options);
            }
        } else {
            newton(f, *opts.gradient_func, *opts.hessian_func, x, options);
        }
        opts.num_iters_returned = options.num_iters_returned;
    } else if (opts.solver == SGD) { // using stochastic gradient descent
        gd_opts options;
        options.err = opts.tolerance;
        options.max_iter = opts.max_iter;
        options.stochastic_batch_size = opts.stochastic_batch_size;
        if (opts.indexed_gradient_func == nullptr) { // no gradient provided
            std::cerr << "minimize_unc() warning: using FD gradients may be very slow!" << std::endl;
            auto df = [f,opts](const arma::vec& u, int ind) -> double {
                auto ff = [f,u,ind](double t) -> double {
                    arma::vec y = u;
                    y(ind) = t;
                    return f(y);
                };
                return deriv(ff,u(ind),opts.tolerance);
            };
            options.max_iter = no_grad_max_iter;
            sgd(df, x, options);
        } else {
            sgd(*opts.indexed_gradient_func, x, options);
        }
        opts.num_iters_returned = options.num_iters_returned;
    } else if (opts.solver == NLCGD) { // using nonlinear conjugate gradient descent
        nonlin_opts options;
        options.max_iter = opts.max_iter;
        options.err = opts.tolerance;
        options.init_jacobian = opts.init_hessian;
        options.use_FD_jacobian = opts.use_FD_hessian;
        options.jacobian_func = opts.hessian_func;
        if (opts.gradient_func == nullptr) { // no gradient function
            std::cerr << "minimize_unc() warning: using FD gradients may be very slow!" << std::endl;
            options.max_iter = no_grad_max_iter;
            auto df = [f,opts](const arma::vec& u) -> arma::vec {return grad(f,u,opts.tolerance);};
            nlcgd(df, x, options);
        } else {
            nlcgd(*opts.gradient_func, x, options);
        }
        opts.num_iters_returned = options.num_iters_returned;
    }

    return f(x);
}

double numerics::minimize_unc(const vec_dfunc& f, arma::vec& x) {
    optim_opts opts;
    return minimize_unc(f, x, opts);
}