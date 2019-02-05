#include "numerics.hpp"

/* MGD : momentum gradient descent.
 * --- f : gradient.
 * --- x : initial guess.
 * --- opts: parameter options. */
void numerics::mgd(const vector_func& f, arma::vec& x, gd_opts& opts) {
    double beta = opts.damping_param;

    arma::vec p = f(x);
    double r = arma::norm(p,"inf");
    double alpha = line_min( [&p,&x,&f,r](double a) -> double {arma::vec q = (-1.0/r)*p; return arma::dot(q,f(x + a*q));} );
    x += (-alpha/r)*p;

    size_t k = 1;
    while (std::abs(alpha*r) > opts.err) {
        if (k >= opts.max_iter) { // too many iterations
            std::cerr << "mgd() error: too many iterations needed for convegence to a root." << std::endl
                      << "returning current best result." << std::endl
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "||f(x)|| = " << arma::norm(f(x), "inf") << " > 0" << std::endl << std::endl;
            opts.num_iters_returned = k;
            return;
        }
        p = beta*p + f(x);
        r = arma::norm(p,"inf");
        alpha = line_min( [&p,&x,&f,r](double a) -> double {arma::vec q = (-1.0/r)*p; return arma::dot(q,f(x + a*q));} );
        x += (-alpha/r)*p;
        k++;
    }
    opts.num_iters_returned = k;
}

/* MGD : momentum gradient descent.
 * --- f : gradient.
 * --- x : initial guess. */
numerics::gd_opts numerics::mgd(const vector_func& f, arma::vec& x) {
    gd_opts opts;
    mgd(f,x,opts);
    return opts;
}