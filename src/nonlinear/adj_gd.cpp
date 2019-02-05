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