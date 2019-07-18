#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o bvp bvp_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

using namespace numerics::ode;
typedef std::vector<double> ddvec;

const double a = -M_PI, b = M_PI;
const uint N = 64;

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

    auto bc = [](const arma::rowvec& uL, const arma::rowvec& uR) -> arma::vec {
        arma::vec v(2);
        v(0) = uL(0) - 1; // solution fixed as 1 at end points
        v(1) = uR(0) - 1;
        return v;
    };

    auto guess = [](double x) -> arma::rowvec {
        arma::rowvec y(2);
        y(0) = 1;
        y(1) = 0;
        return y;
    };
    
    arma::vec x;
    arma::mat U;

    bvp_k bvp_solver(4);
        // bvp_solver.max_iterations = 20;
        // bvp_solver.tol = 1e-7;
        x = arma::linspace(a,b,N);
        U = arma::zeros(N,2);
        for (uint i=0; i < N; ++i) U.row(i) = guess(x(i));
        bvp_solver.ode_solve(x,U,f,bc);
        // bvp_solver.ode_solve(x,U,f,J,bc);

    /* bvp_cheb bvp_solver;
        bvp_solver.num_pts = N;
        // bvp_solver.max_iterations = 20;
        // bvp_solver.tol = 1e-7;
        x = {a,b};
        bvp_solver.ode_solve(x,U,f,bc,guess);
        // bvp_solver.ode_solve(x,U,f,J,bc,guess); */

    /* bvpIIIa bvp_solver;
        // bvp_solver.max_iterations = 20;
        // bvp_solver.tol = 1e-7;
        x = arma::linspace(a,b,N);
        U = arma::zeros(N,2);
        for (uint i=0; i < N; ++i) U.row(i) = guess(x(i));
        bvp_solver.ode_solve(x,U,f,bc);
        // bvp_solver.ode_solve(x,U,f,J,bc); */


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