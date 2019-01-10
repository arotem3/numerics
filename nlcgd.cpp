#include "numerics.hpp"

//--- nonlinear conjugate gradient method ---//
//----- f : gradient function ---------------//
//----- x : guess, and solution -------------//
void numerics::nlcgd(const vector_func& f, arma::vec& x, nonlin_opts& opts) {
    int n = x.n_elem;
    
    arma::vec p;
    arma::vec prev_x;
    double alpha;
    double r;

    size_t k = 0;
    do {
        if (k >= opts.max_iter) { // too many iterations
            std::cerr << "\nnlcgd() failed: too many iterations needed for convergence." << std::endl
                 << "returning current best estimate." << std::endl
                 << "!!!---not necessarily a good estimate---!!!" << std::endl
                 << "||f(x)|| = " << arma::norm(p, "inf") << " > 0" << std::endl << std::endl;
            break;
        }
        if (2*n-4 == 0) prev_x = x;
        for (int i(1); i <= 2*n-2; ++i) {
            p = -f(x);
            r = arma::norm(p,"inf");
            alpha = line_min( [&p,&x,&f,r](double a) -> double {return arma::dot(p/r,f(x + (a/r)*p));} );
            x += (alpha/r)*p;
            k++;
            if (i == 2*n - 4) prev_x = x;
        }
        p = prev_x - x;
        r = arma::norm(p,"inf");
        alpha = line_min( [&p,&x,&f,r](double a) -> double {return arma::dot(p/r,f(x + (a/r)*p));} );
        x += (alpha/r)*p;
        k++;
    } while (std::abs(alpha*r) > opts.err);

    opts.num_iters_returned = k;
}

numerics::nonlin_opts numerics::nlcgd(const vector_func& f, arma::vec& x) {
    nonlin_opts opts;
    nlcgd(f,x,opts);
    return opts;
}