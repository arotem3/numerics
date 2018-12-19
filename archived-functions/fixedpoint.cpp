#include "numerics.hpp"

//--- solves the fixed point iteration problem U = f(U) ---//
//----- f  : function of U --------------------------------//
//----- U0 : initial guess for U --------------------------//
arma::rowvec numerics::FPI(std::function<arma::rowvec(const arma::rowvec&)> f, const arma::rowvec& U0, nonlin_opts& opts) {
    double convergence = INFINITY;
    arma::rowvec U = U0;
    arma::rowvec U1;
    int counter = 0;

    while ( convergence > opts.err ) {
        U1 = U;
        U = f(U);
        convergence = arma::norm(U1 - U);
        counter++;
        if (counter > opts.max_iter) {
            std::cerr << "warning in FPI(): could not converge bellow error requirements for the given number of iterations. Returning current best approximation." << std::endl;
            break;
        }
    }

    opts.num_iters_returned = counter;

    return U;
}

arma::rowvec numerics::FPI(std::function<arma::rowvec(const arma::rowvec&)> f, const arma::rowvec& U0) {
    nonlin_opts opts;
    opts.err = root_err;
    opts.max_iter = 400;
    
    return FPI(f,U0,opts);
}

//--- univariate version ---//
double numerics::FPI(std::function<double(double)> f, double U0, int maxIter, double err) {
    double convergence = INFINITY;
    double U = U0;
    double U1;
    int counter = 0;

    while ( convergence > err ) {
        U1 = U;
        U = f(U);
        convergence = std::abs(U1 - U);
        counter++;
        if (counter > maxIter) {
            std::cerr << "warning in FPI(): could not converge bellow error requirements for the given number of iterations. Returning current best approximation." << std::endl;
            break;
        }
    }

    return U;
}