#include "numerics.hpp"

//--- momentum gradient descent ---//
//----- f : gradient --------------//
//----- x : initial guess ---------//
//----- opts: parameter options ---//
void numerics::mgd(const vector_func& f, arma::vec& x, gd_opts& opts) {
    double gamma = opts.damping_param;
    double alpha = opts.step_size;

    arma::vec p = f(x);

    size_t k = 0;
    while (arma::norm(p,"inf") > opts.err) {
        if (k > opts.max_iter) { // too many iterations
            std::cerr << "mgd() error: too many iterations needed for convegence to a root." << std::endl
                      << "returning current best result." << std::endl
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "||f(x)|| = " << arma::norm(f(x), "inf") << " > 0" << std::endl << std::endl;
            opts.num_iters_returned = k;
            return;
        }
        x -= alpha*p;
        p = gamma*p + f(x);
        k++;
    }
    opts.num_iters_returned = k;
}

numerics::gd_opts numerics::mgd(const vector_func& f, arma::vec& x) {
    gd_opts opts;
    mgd(f,x,opts);
    return opts;
}