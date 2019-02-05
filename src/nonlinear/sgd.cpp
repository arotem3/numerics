#include "numerics.hpp"

/* SGD : stochastic batch gradient descent.
 * --- f : gradient function.
 * --- x : initial guess.
 * --- opts: options struct. */
void numerics::sgd(const sp_vector_func& f, arma::vec& x, gd_opts& opts) {
    int n = x.n_elem;
    int bs = opts.stochastic_batch_size;

    double alpha;
    double r;
    arma::vec p;
    
    size_t k = 0;
    do {
        if (k > opts.max_iter) { // too many iterations
            std::cerr << "sgd() error: too many iterations needed for convegence to a root." << std::endl
                      << "returning current best result." << std::endl
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "||f(x)|| = " << arma::norm(p, "inf") << " > 0" << std::endl << std::endl;
            opts.num_iters_returned = k;
            return;
        }
        p = arma::zeros(n);
        arma::uvec ind = arma::shuffle( arma::regspace<arma::uvec>(0,n-1) );
        ind = ind( arma::span(0,bs-1) );
        for (int i(0); i < bs; ++i) {
            p( ind(i) ) = f( x, ind(i) );
        }
        r = arma::norm(p,"inf");
        alpha = line_min(
            [&f,&ind,&p,&x,n,bs,r](double a) -> double {
                arma::vec z = arma::zeros(n);
                arma::vec q = (-1.0/r)*p;
                for (int i(0); i < bs; ++i) {
                    z( ind(i) ) = f(x + a*q, ind(i));
                }
                return arma::dot(q,z);
            }
        );
        x += (-alpha/r)*p;
        k++;
    } while (std::abs(alpha*r) > opts.err);
    opts.num_iters_returned = k;
}

/* SGD : stochastic batch gradient descent.
 * --- f : gradient function.
 * --- x : initial guess. */
numerics::gd_opts numerics::sgd(const sp_vector_func& f, arma::vec& x) {
    gd_opts opts;
    opts.damping_param = 0;
    sgd(f, x, opts);
    return opts;
}