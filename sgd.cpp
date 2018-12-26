#include "numerics.hpp"

//--- stochastic gradient descent ---//
//----- f : gradient function -------//
//----- x : initial guess -----------//
//----- opts: options struct --------//
void numerics::sgd(const sp_vector_func& f, arma::vec& x, gd_opts& opts) {
    int n = x.n_elem;
    int bs = opts.stochastic_batch_size;
    double alpha = opts.step_size;
    double gamma = opts.damping_param;

    arma::vec p = arma::zeros(n);
    arma::uvec ind = arma::shuffle( arma::regspace<arma::uvec>(0,n-1) );
    ind = ind( arma::span(0,bs-1) );
    for (int i(0); i < bs; ++i) {
        p( ind(i) ) = f( x, ind(i) );
    }
    
    size_t k = 0;
    while (arma::norm(p,"inf") > opts.err) {
        if (k > opts.max_iter) { // too many iterations
            std::cerr << "sgd() error: too many iterations needed for convegence to a root." << std::endl
                      << "returning current best result." << std::endl
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "||f(x)|| = " << arma::norm(p, "inf") << " > 0" << std::endl << std::endl;
            opts.num_iters_returned = k;
            return;
        }
        x -= alpha*p;

        arma::vec g = arma::zeros(n);
        ind = arma::shuffle( arma::regspace<arma::uvec>(0,n-1) );
        ind = ind( arma::span(0,bs-1) );
        for (int i(0); i < bs; ++i) {
            g( ind(i) ) = f( x, ind(i) );
        }
        p = gamma*p + g;
        k++;
    }
    opts.num_iters_returned = k;
}

numerics::gd_opts numerics::sgd(const sp_vector_func& f, arma::vec& x) {
    gd_opts opts;
    opts.damping_param = 0;
    sgd(f, x, opts);
    return opts;
}