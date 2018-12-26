#include "numerics.hpp"

double lineMin(std::function<arma::vec(const arma::vec&)>, arma::vec&, arma::vec&);

//--- stochastic gradient descent algorithm for unconstrained local optimization of multivariate function ---//
//----- f  : double = f(x) our function to optimized --------------------------------------------------------//
//----- df : vec = df(x) gradient of our function -----------------------------------------------------------//
//----- x0 : initial guess close to a local minimum, will also be where optimum vector is stored ------------//
double numerics::sgd(std::function<double(const arma::vec&)> f, std::function<arma::vec(const arma::vec&)> df, arma::vec& x0) {
    double err = root_err;
    int n = x0.n_elem;
    arma::vec p = -arma::normalise(df(x0)) + 0.1*arma::normalise(arma::randu(n));
    double alpha = lineMin(df,x0,p);
    arma::vec x1 = x0 + alpha*p;
    short k = 1;
    while (arma::norm(df(x1)) > err) {
        if (k > sgd_max_iter) {
            std::cerr << "\nsgd() failed: too many iterations needed to converge." << std::endl
                      << "returning current best estimate." << std::endl
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "gradient norm = " << arma::norm(df(x1)) << " > 0" << std::endl << std::endl;
            x0 = x1;
            return f(x0);
        }
        x0 = x1;
        p = -arma::normalise(df(x0)) ;
        if (k < 0.9*sgd_max_iter) {
            p += 0.1*arma::normalise(arma::randu(n))-0.05;
        }
        alpha = lineMin(df,x0,p);
        x1 = x0 + alpha*p;
        k++;
    }
    x0 = x1;
    return f(x0);
}

double numerics::sgd(std::function<double(const arma::vec&)> f, arma::vec& x0) {
    auto df = [&f](const arma::vec& x) { return grad(f,x); };
    
    double err = root_err;
    int n = x0.n_elem;
    arma::vec p = -arma::normalise(df(x0)) + 0.1*arma::normalise(arma::randu(n))-0.05;
    double alpha = lineMin(df,x0,p);;
    arma::vec x1 = x0 + alpha*p;
    short k = 1;
    while (arma::norm(df(x1)) > err) {
        if (k > no_grad_max_iter) {
            std::cerr << "\nsgd() failed: too many iterations needed to converge." << std::endl
                      << "returning current best estimate." << std::endl
                      << "!!!---not necessarily a good estimate---!!!" << std::endl
                      << "gradient norm = " << arma::norm(df(x1)) << " > 0" << std::endl << std::endl;
            x0 = x1;
            return f(x0);
        }
        x0 = x1;
        p = -arma::normalise(df(x0));
        if (k < 0.9*no_grad_max_iter) {
            p += 0.1*arma::normalise(arma::randu(n))-0.05;
        }
        alpha = lineMin(df,x0,p);
        x1 = x0 + alpha*p;
        k++;
    }
    x0 = x1;
    return f(x0);
}

//---secant method specifically to evaluate the minimum for our line search
double lineMin(std::function<arma::vec(const arma::vec&)> df, arma::vec& x, arma::vec& p) {
    double err = numerics::root_err;
    auto h = [&](double a) { // h(a) = p^T * df(x + a*p)
        return arma::dot(p,df(x + a*p));
    };
    auto nextP = [&h](double a1, double a2) {
        double numer = h(a2) * (a2 - a1);
        double denom = h(a2) - h(a1);
        return (a2 - numer/denom);
    };

    //(1) attempt bisection method on interval [0.01,20]
    double a1 = 0.01;
    double a2 = 20;
    //(1.a) check if endpoints are good minima
    if (std::abs(h(a1)) < err) {
        return a1;
    } else if (std::abs(h(a2)) < err) {
        return a2;
    }
    //(1.b) bisection method loop
    double midpt;
    while (std::abs(a2 - a1) > 0.5) {
        midpt = (a2 + a1)/2;
        if (std::abs(h(a1) * h(midpt)) < err) { // check if p is a good minimum
            return midpt;
        } else if (h(a1) * h(midpt) < 0) {
            a2 = midpt;
        } else {
            a1 = midpt;
        }
    }

    midpt = (a2 + a1)/2;
    if (std::abs(h(a1) - h(midpt)) > std::abs(h(a2) - h(midpt))) a2 = midpt;
    else a1 = midpt;

    //(2) finish off convergence with secant method
    short k = 1;
    while ( std::abs(a2 - a1) > err && a2 < 20 && a2 > 0.01 ) {
        //---check if line search takes too long.
        if (k > 200) {
            std::cerr << "lineMin() failed: too many iterations neededf for convergence. returning current best estimate." << std::endl
                 << "!!!---not necessarily a good estimate---!!!" << std::endl
                 << "p^T * df(x + a*p) = " << arma::dot(df(x + a2*p),p) << " > 0 for this 'a' val." << std::endl;
            return a2;
        }

        double z = nextP(a1,a2);
        a1 = a2;
        a2 = z;
        k++;
    }
    //(3) we do not want to permit too large or too small alphas
    if (a2 > 20) {
        std::cerr << "lineMin() warning: initial point 'x' is potentially too far from minimum for lineMin() to be effective." << std::endl;
        a2 = 20;
    }
    if (a2 < 0.01) {
        std::cerr << "lineMin() warning: initial point 'x' is potentially too close to minimum for lineMin() to be effective." << std::endl;
        a2 = 0.01;
    }
    return a2;
}