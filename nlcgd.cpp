#include "numerics.hpp"

//--- nonlinear conjugate gradient method ---//
//----- f : gradient function ---------------//
//----- x : guess, and solution -------------//
void numerics::nlcgd(const vector_func& f, arma::vec& x, nonlin_opts& opts) {
    int n = x.n_elem;
    
    arma::vec p = -f(x);
    arma::vec s;
    double alpha, r;

    size_t k = 0;
    do {
        if (k >= opts.max_iter) { // too many iterations
            std::cerr << "\nnlcgd() failed: too many iterations needed for convergence." << std::endl
                 << "returning current best estimate." << std::endl
                 << "!!!---not necessarily a good estimate---!!!" << std::endl
                 << "||f(x)|| = " << arma::norm(p, "inf") << " > 0" << std::endl << std::endl;
            break;
        }
        r = 1/arma::norm(p, "inf");
        alpha = line_min( [&p,&x,&f,r](double a) -> double {return r*arma::dot(p,f(x + (a*r)*p));} );
        double sts = arma::norm(s,2);
        arma::vec ds = (alpha*r)*p - s;

        s = (alpha*r)*p;
        if (s.has_nan()) {
            std::cerr << "nlcgd() warning: encountered NAN before convergence." << std::endl
                      << "returning current best estimate." << std::endl
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "||f(x)|| = " << arma::norm(p, "inf") << " > 0" << std::endl << std::endl;
            break;
        }
        x += s;
        k++;
        if (k%n == 0) p = -f(x);
        else {
            double beta = std::max(arma::dot(s,ds)/sts, 0.0);
            p = -f(x) + beta*p;
        }
    } while (std::abs(alpha*r) > opts.err);

    opts.num_iters_returned = k;
}

numerics::nonlin_opts numerics::nlcgd(const vector_func& f, arma::vec& x) {
    nonlin_opts opts;
    nlcgd(f,x,opts);
    return opts;
}