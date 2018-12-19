#include "numerics.hpp"

arma::vec binarr(int, int);

//--- monte carlo integration method for arbitrary dimension, bounded in a box ---//
//----- f  : function to integrate -----------------------------------------------//
//----- a  : lower bounds of box -------------------------------------------------//
//----- b  : upper bounds of box -------------------------------------------------//
//----- err: estimate of upper bound on error of the integral --------------------//
//----- N  : number of sample points ---------------------------------------------//
double numerics::mcIntegrate(std::function<double(const arma::vec&)> f, const arma::vec& a, const arma::vec& b, double err, int N) {
    arma::arma_rng::set_seed_random();
    double I = 0;
    int dim = a.n_elem;
    // (0.a) for the extreme recursive case where N --> 1 return the evaluation of the function at a random point
    if (N == 1) {
        arma::vec X = (b - a)*arma::randu() + a;
        I = arma::prod(b-a) * f(X);
        //std::cerr << "warning in mcIntegrate(): recursive base case reached! No further error control possible." << std::endl;
        return I;
    }

    // (0.b) auxilary calculations
    int L = std::pow(2,dim); //         L = 2^dim = number of strata
    arma::vec range = (b - a)/2; //     range = range of a to b divided by 2 for convinience
    int M = N/L; //                     points per strata

    if (M == 0) { // recursion too deep!
        arma::vec X = (b-a)*arma::randu() + a;
        I = arma::prod(b-a) * f(X);
        //std::cerr << "warning in mcIntegrate(): recursive base case reached! No further error control possible." << std::endl;
        return I;
    }

    // (1) evaluate the function over all [a,b] to approximate integral
    arma::mat X1 = arma::randu(dim,N);
    X1.each_col() %= (b - a);
    X1.each_col() += a; //              X1 = (b - a)*rand + a
    arma::vec F1 = eval(f,X1); //       F1 = f(X1)
    double I1 = arma::prod(b-a)/N * arma::accu(F1); //  integral approx
    double v1 = arma::var(F1); //                       variance of calculations

    // (2) evaluate at L stata
    arma::vec F2(M*L,arma::fill::zeros);
    for (int n(0); n < L; ++n) {
        arma::mat X2 = arma::randu(dim,M);
        X2.each_col() %= range;
        X2.each_col() += a + (binarr(n,dim) % range);// random values in the range of a given strata
        F2(arma::span(n*M,(n+1)*M-1)) = eval(f,X2);
    }
    double I2 = arma::prod(b - a)/N * arma::sum(F2);
    double v2 = arma::var(F2);
    

    // (3) compare the two integrals
    double tscr = (I1 - I2)/std::sqrt((v1 + v2)/N);
    if (std::abs(I1 - I2) < err && std::abs(tscr) < 0.0627) { // we estimate that our value is within 5% of the true value
        I = I1;
    } else { // our estimate is not good enough restratify
        I = 0;
        for (int n(0); n < L; ++n) {
            arma::vec u = binarr(n,dim) % range;
            arma::vec lower = a + u;
            arma::vec upper = a + u + range;
            I = I + mcIntegrate(f,lower,upper,err/L,M);
        }
    }
    return I;
}

arma::vec binarr(int n, int dim) {
    arma::vec v(dim, arma::fill::zeros);
    for (int i(0); i < dim; ++i) {
        if (n%2 == 1) {
            v(i) = 1;
        }
        n = n/2; // integer division
    }
    return v;
}