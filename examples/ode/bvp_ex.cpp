#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o bvp bvp_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

typedef std::vector<double> dvec;

int main() {
    double a = -M_PI, b = M_PI;
    int N = 32;

    std::cout << "Now we will solve the nonlinear boundary value problem:" << std::endl
              << "\tu' = v" << std::endl << "\tv' = 2(1 - x^2)*u" << std::endl
              << "\t-pi < x < pi" << std::endl
              << "\tu(-pi) = 1\tu(pi) = 1" << std::endl
              << "we will use an initial guess of u(x) = 1 and v(x) = 0" << std::endl
              << std::endl << std::endl;

    auto f = [](double x, const arma::vec& u) -> arma::vec {
        arma::vec up(2);
        up(0) = u(1);
        up(1) = 2*(1 - x*x)*u(0);
        return up;
    };

    auto J = [](double x,const arma::vec& u) -> arma::mat {
        arma::mat A = arma::zeros(2,2);
        A(0,1) = 1;
        A(1,0) = 2*(1 - x*x);
        return A;
    };

    auto bc = [](const arma::vec& uL, const arma::vec& uR) -> arma::vec {
        arma::vec v(2);
        v(0) = uL(0) - 1; // solution fixed as 1 at end points
        v(1) = uR(0) - 1;
        return v;
    };

    auto guess = [](double x) -> arma::vec {
        arma::vec y(2);
        y(0) = 1;
        y(1) = 0;
        return y;
    };
    
    arma::vec x = arma::linspace(a,b,N);
    arma::mat U(2,N);
    for (int i=0; i < N; ++i) U.col(i) = guess(x(i)); // set U to the initial guess

    // numerics::ode::BVPk bvp_solver(4);
    numerics::ode::BVPCheb bvp_solver(N);
    // numerics::ode::BVP3a bvp_solver;

    numerics::ode::ODESolution sol = bvp_solver.ode_solve(f,bc,x,U);
    // numerics::ode::ODESolution sol = bvp_solver.ode_solve(f,J,bc,x,U);

    // an effective way to interpolate the numerical output of these bvp methods is via hermite splines
    numerics::HSplineInterp soln;
    arma::mat dU(arma::size(sol.solution));
    for (int i=0; i < N; ++i) dU.row(i) = f(sol.t(i),sol.solvec.at(i)).t();
    soln.fit(sol.t, sol.solution, dU);

    // if you are using bvp_cheb consider using a PolyInterp object instead...
    // numerics::PolyInterp soln;
    // soln.fit(sol.t, sol.solution);

    arma::vec xx = arma::linspace(a,b,500);
    arma::mat uu = soln.predict(xx);

    std::cout << "Number of nonlinear iterations needed by solver: " << bvp_solver.num_iter << std::endl;
    dvec x1 = arma::conv_to<dvec>::from(sol.t);
    dvec u1 = arma::conv_to<dvec>::from(sol.solution.col(0));
    dvec v1 = arma::conv_to<dvec>::from(sol.solution.col(1));
    dvec x2 = arma::conv_to<dvec>::from(xx.col(0));
    dvec u2 = arma::conv_to<dvec>::from(uu.col(0));
    dvec v2 = arma::conv_to<dvec>::from(uu.col(1));
    
    matplotlibcpp::subplot(2,1,1);
    std::map<std::string,std::string> ls = {{"marker","o"},{"label","u(x)"},{"ls","none"},{"color","purple"}};
    matplotlibcpp::plot(x1,u1,ls);
    matplotlibcpp::plot(x2,u2,"-b");
    matplotlibcpp::legend();
    matplotlibcpp::subplot(2,1,2);
    ls["label"] = "v(x)"; ls["color"] = "orange"; ls["marker"] = "o"; ls["ls"] = "none";
    matplotlibcpp::plot(x1,v1,ls);
    matplotlibcpp::plot(x2,v2,"-r");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}