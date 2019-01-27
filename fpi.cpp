#include "numerics.hpp"

void numerics::mix_fpi(const vector_func& g, arma::vec& x, fpi_opts& opts) {
    int n = x.n_elem;
    unsigned int m = opts.steps_to_remember;
    cyc_queue X(n, m);
    cyc_queue G(n, m);

    unsigned int k = 1;
    arma::vec f;

    do {
        if (k >= opts.max_iter) {
            std::cerr << "mix_fpi() error: too many iterations needed for convegence to a root." << std::endl
                      << "returning current best result." << std::endl
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "||x - g(x)|| = " << arma::norm(g(x)-x, "inf") << " > 0" << std::endl << std::endl;
            break;
        }

        f = g(x);
        X.push(x);
        G.push(f);

        int mk = std::min(k,m);
        arma::mat FF = arma::join_cols(X.data() - G.data(), arma::ones(1,mk));
        arma::vec b = arma::zeros(n+1);
        b(n) = 1;

        x = G.data() * arma::solve(FF,b);
        k++;

    } while (arma::norm(f - x,"inf") > opts.err);
    opts.num_iters_returned = k;
}

numerics::fpi_opts numerics::mix_fpi(const vector_func& g, arma::vec& x) {
    fpi_opts opts;
    mix_fpi(g,x,opts);
    return opts;
}