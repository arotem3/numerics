#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o bvp bvp_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

using namespace numerics::ode;
typedef std::vector<double> ddvec;

int main() {
    std::cout << "Now we will solve the nonlinear boundary value problem:" << std::endl
              << "\tu' = v" << std::endl << "\tv' = 2(1 - x^2)*u" << std::endl
              << "\t-pi < x < pi" << std::endl
              << "\tu(-pi) = 1\tu(pi) = 1" << std::endl
              << "we will use an initial guess of u(x) = 1 and v(x) = 0" << std::endl
              << std::endl << std::endl;

    auto f = [](double x, const arma::rowvec& u) -> arma::rowvec {
        arma::rowvec up(2, arma::fill::zeros);
        up(0) = u(1);
        up(1) = 2*(1 - x*x)*u(0);
        return up;
    };

    auto J = [](double x,const arma::rowvec& u) -> arma::mat {
        arma::mat A = arma::zeros(2,2);
        A(0,1) = 1;
        A(1,0) = 2*(1 - x*x);
        return A;
    };

    boundary_conditions bc;
    bc.xL = -M_PI;
    bc.xR = M_PI;
    bc.condition = [](const arma::rowvec& uL, const arma::rowvec& uR) -> arma::rowvec {
        arma::rowvec v(2);
        v(0) = uL(0) - 1; // solution fixed as 1 at end points
        v(1) = uR(0) - 1;
        return v;
    };

    auto guess = [](const arma::vec& x) -> arma::mat {
        arma::mat y = arma::zeros(x.n_elem,2);
        y.col(0) = arma::ones(x.n_elem);
        y.col(1) = arma::zeros(x.n_elem);
        return y;
    };

    bvp bvp_solver;
    // bvp_solver.max_iterations = 1000;
    // bvp_solver.tol = 1e-7; // play around with the nonlinear solver tolerance
    // bvp_solver.order = bvp_solvers::SECOND_ORDER; std::cout << "using second order solver..." << std::endl;
    // bvp_solver.order = bvp_solvers::FOURTH_ORDER; std::cout << "using fourth order solver..." << std::endl;
    bvp_solver.order = bvp_solvers::CHEBYSHEV; std::cout << "using spectral solver..." << std::endl;
    bvp_solver.num_points = 64;

    arma::vec x;
    arma::mat U;
    // bvp_solver.ode_solve(x,U,f,bc,guess); 
    bvp_solver.ode_solve(x,U,f,J,bc,guess); // providing a jacobian function can improve performance
    std::cout << "Number of nonlinear iterations needed by solver: " << bvp_solver.num_iterations() << std::endl;
            ddvec x1 = arma::conv_to<ddvec>::from(x);
            ddvec u1 = arma::conv_to<ddvec>::from(U.col(0));
            ddvec v1 = arma::conv_to<ddvec>::from(U.col(1));
            
            matplotlibcpp::subplot(2,1,1);
            std::map<std::string,std::string> ls = {{"marker","o"},{"label","u(x)"},{"ls","-"},{"color","blue"}};
            matplotlibcpp::plot(x1,u1,ls);
            matplotlibcpp::legend();
            matplotlibcpp::subplot(2,1,2);
            ls["label"] = "v(x)"; ls["color"] = "red";
            matplotlibcpp::plot(x1,v1,ls);
            matplotlibcpp::legend();
            matplotlibcpp::show();

    return 0;
}