#include "numerics.hpp"

/* ADJ_GD : "adjusted" gradient descent, an acceleration method of my own desing
 * --- f : gradient function
 * --- x : intial guess and output variable
 * --- opts : options for solver */
void numerics::adj_gd(const vector_func& f, arma::vec& x, nonlin_opts& opts) {
    double alpha, r;
    unsigned int k = 1;
    arma::vec p, prev_x;

    do {
        if (k >= opts.max_iter) { // error
            std::cerr << "\nadj_gd() failed: too many iterations needed for convergence." << std::endl
                 << "returning current best estimate." << std::endl
                 << "!!!---not necessarily a good estimate---!!!" << std::endl
                 << "||f(x)|| = " << arma::norm(p, "inf") << " > 0" << std::endl << std::endl;
            break;
        }

        if (k%3==1 || k%3==2) {
            p = -f(x);
            if (k%3==1) prev_x = x;
        } else { // every third itteration
            p = prev_x - x;
        }
        r = arma::norm(p,"inf");
        alpha = line_min( [&p,&x,&f,r](double a) -> double {return arma::dot(p,f(x + (a/r)*p))/r;} );
        x += (alpha/r)*p;
        k++;
    } while (std::abs(alpha*r) > opts.err);

    opts.num_iters_returned = k;
}

/* ADJ_GD : "adjusted" gradient descent, an acceleration method of my own desing
 * --- f : gradient function
 * --- x : intial guess and output variable */
numerics::nonlin_opts numerics::adj_gd(const vector_func& f, arma::vec& x) {
    nonlin_opts opts;
    adj_gd(f,x,opts);
    return opts;
}

/* LINEAR_ADJ_GD : like adj_gd by for linear systems A*x = b
 * --- A, b, x : A*x = b
 * --- opts : options for solver */
void numerics::linear_adj_gd(arma::mat& A, arma::mat& b, arma::mat& x, cg_opts& opts) {
    if (!opts.is_symmetric || !A.is_symmetric()) {
        std::cerr << "linear_adj_gd() warning: A is not symmetric. Setting A = A.t()*A and b =  A.t()*b." << std::endl;

        b = A.t()*b;
        A = A.t()*A;
    }
    if (opts.max_iter == 0) opts.max_iter = b.n_rows;
    if (x.empty()) x = arma::randn(arma::size(b));

    int k = 1;
    arma::mat g, v, prev_x;
    do {
        if (k >= opts.max_iter) {
            std::cerr << "linear_adj_gd() failed: too many iterations needed for convergence." << std::endl
                 << "returning current best estimate." << std::endl
                 << "!!!---not necessarily a good estimate---!!!" << std::endl
                 << "||f(x)|| = " << arma::norm(v,"fro") << " > 0" << std::endl << std::endl;
            break;
        }
        if (k%3==1 || k%3==2) {
            g = A*x - b;
            v = g;
            if (k%3==1) prev_x = x;
        } else { // every third itteration
            g = prev_x - x;
            v = A*x - b;
        }
        arma::rowvec alpha(b.n_cols);
        for (int i=0; i < b.n_cols; ++i) {
            alpha(i) = (arma::dot(b.col(i),v.col(i)) - arma::dot(x.col(i),A*v.col(i))) / arma::dot(g.col(i),A*v.col(i));
        }
        if (k%3==0) {
            // since we are using a multistep method, we need to check stability. stability polynomial = (r-1)*(r^2 + alpha(r+1))
            for (int i=0; i < b.n_cols; ++i) {
                if (alpha(i) > -0.5) {
                    x.col(i) += alpha(i) * g.col(i);
                    k++;
                }
            }
        } else {
            for (int i=0; i < b.n_cols; ++i) {
                x.col(i) += alpha(i) * g.col(i);
                k++;
            }
        }
    } while (arma::norm(v,"fro") > opts.err);
    opts.num_iters_returned = k;
}


numerics::cg_opts numerics::linear_adj_gd(arma::mat& A, arma::mat& b, arma::mat& x) {
    cg_opts opts;
    linear_adj_gd(A,b,x,opts);
    return opts;
}

void numerics::linear_adj_gd(const arma::sp_mat& A, const arma::mat& b, arma::mat& x, cg_opts& opts) {
    if (!A.is_symmetric()) {
        std::cerr << "linear_adj_gd() error: sparse A is not symmetric. exiting." << std::endl;
        return;
    }
    if (opts.max_iter == 0) opts.max_iter = b.n_rows;
    if (x.empty()) x = arma::randn(arma::size(b));

    int k = 1;
    arma::mat g, v, prev_x;
    do {
        if (k >= opts.max_iter) {
            std::cerr << "linear_adj_gd() failed: too many iterations needed for convergence." << std::endl
                 << "returning current best estimate." << std::endl
                 << "!!!---not necessarily a good estimate---!!!" << std::endl
                 << "||f(x)|| = " << arma::norm(v,"fro") << " > 0" << std::endl << std::endl;
            break;
        }
        if (k%3==1 || k%3==2) {
            g = A*x - b;
            v = g;
            if (k%3==1) prev_x = x;
        } else { // every third itteration
            g = prev_x - x;
            v = A*x - b;
        }
        arma::rowvec alpha(b.n_cols);
        for (int i=0; i < b.n_cols; ++i) {
            alpha(i) = (arma::dot(b.col(i),v.col(i)) - arma::dot(x.col(i),A*v.col(i))) / arma::dot(g.col(i),A*v.col(i));
        }
        if (k%3==0) {
            // since we are using a multistep method, we need to check stability. stability polynomial = (r-1)*(r^2 + alpha(r+1))
            for (int i=0; i < b.n_cols; ++i) {
                if (alpha(i) > -0.5) {
                    x.col(i) += alpha(i) * g.col(i);
                    k++;
                }
            }
        } else {
            for (int i=0; i < b.n_cols; ++i) {
                x.col(i) += alpha(i) * g.col(i);
                k++;
            }
        }
    } while (arma::norm(v,"fro") > opts.err);
    opts.num_iters_returned = k;
}

numerics::cg_opts numerics::linear_adj_gd(const arma::sp_mat& A, const arma::mat& b, arma::mat& x) {
    cg_opts opts;
    linear_adj_gd(A,b,x,opts);
    return opts;
}