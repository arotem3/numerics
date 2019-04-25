#include "ODE.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o bvp examples/bvp_ex.cpp examples/wait.cpp -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

using namespace ODE;
namespace plt = matplotlibcpp;
typedef std::vector<double> ddvec;

int main() {
    std::cout << "Now we will solve the nonlinear boundary value problem:" << std::endl
              << "\tu' = v" << std::endl << "\tv' = -sin(u)" << std::endl
              << "\t0 < x < 2*pi" << std::endl
              << "\tu(0) = 1\tu(2*pi) = 1" << std::endl
              << "we will use an initial guess of u(x) = cos(x) and v(x) = sin(x) i.e. the result from linearization" << std::endl
              << "we will also use an intial guess of u(x) = 1 and v(x) = 0 as another example.\n(this will demonstrate non-uniqueness depending on your choice of solver)."
              << std::endl << std::endl;

    odefun f = [](double x, const arma::rowvec& u) -> arma::rowvec {
        arma::rowvec up(2, arma::fill::zeros);
        up(0) = u(1);
        up(1) = -std::sin(u(0));
        return up;
    };

    odeJac J = [](double x,const arma::rowvec& u) -> arma::mat {
        arma::mat A = arma::zeros(2,2);
        A(0,1) = 1;
        A(1,0) = -std::cos(u(0));
        return A;
    };

    bcfun bc;
    bc.xL = 0;
    bc.xR = 2*M_PI;
    bc.func = [](const arma::rowvec& uL, const arma::rowvec& uR) -> arma::rowvec {
        arma::rowvec v(2,arma::fill::zeros);
        v(0) = uL(0) - 1; // solution fixed as 1 at end points
        v(1) = uR(0) - 1;
        return v;
    };

    soln_init guess = [](const arma::vec& x) -> arma::mat {
        arma::mat y = arma::zeros(x.n_elem,2);
        y.col(0) = arma::cos(x);
        y.col(1) = arma::sin(x);
        return y;
    };

    bvp_opts opts;
    opts.nlnopts.err = 1e-15; // play around with the nonlinear solver tolerance
    // opts.order = bvp_solvers::SECOND_ORDER; std::cout << "using second order solver..." << std::endl;
    // opts.order = bvp_solvers::FOURTH_ORDER; std::cout << "using fourth order solver..." << std::endl;
    opts.order = bvp_solvers::CHEBYSHEV; std::cout << "using spectral solver" << std::endl;
    opts.num_points = 50;
    opts.jacobian_func = &J; // providing a jacobian function improves runtime significantly

    dsolnp soln = bvp(f, bc, guess, opts);

    ddvec x = arma::conv_to<ddvec>::from(soln.independent_var_values);
    ddvec u1 = arma::conv_to<ddvec>::from(soln.solution_values.col(0));
    ddvec v1 = arma::conv_to<ddvec>::from(soln.solution_values.col(1));
    
    plt::plot(x,u1,"-ob");
    plt::plot(x,v1,"-or");

    std::cout << "Number of nonlinear iterations needed by solver: " << opts.nlnopts.num_iters_returned << std::endl;

    guess = [](const arma::vec& x) -> arma::mat {
        arma::mat y = arma::zeros(x.n_elem,2);
        y.col(0) = arma::ones(x.n_elem);
        y.col(1) = arma::zeros(x.n_elem);
        return y;
    };

    soln = bvp(f, bc, guess, opts);
    std::cout << "Number of nonlinear iterations needed by solver: " << opts.nlnopts.num_iters_returned << std::endl;

    x = arma::conv_to<ddvec>::from(soln.independent_var_values);
    ddvec u2 = arma::conv_to<ddvec>::from(soln.solution_values.col(0));
    ddvec v2 = arma::conv_to<ddvec>::from(soln.solution_values.col(1));
    plt::plot(x,u2,"--sm");
    plt::plot(x,v2,"--sk");
    plt::show();

    return 0;
}