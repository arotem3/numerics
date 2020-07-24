#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o diffeq ode_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

using namespace numerics::ode;
typedef std::vector<double> ddvec;

const double mu = 100; // bigger value --> more stiff
const double t0 = 0;
const double tf = 2*mu;
const double U0_1 = 2;
const double U0_2 = 0;
const bool add_event = false;

int main() {
    std::cout << "Let's solve the vanderpol equation with mu=" << mu << std::endl
              << "We will assume the true solution is the output of rk45 with error < 1e-6." << std::endl
              << "you can play around with the solvers and their options within the example code." << std::endl << std::endl;

    auto f = [](double t, const arma::rowvec& u) -> arma::rowvec {
        arma::rowvec v(2);
        double x = u(0);
        double y = u(1);
        v(0) = y;
        v(1) = mu*(1-x*x)*y - x; // vanderpol with mu = 1
        return v;
    };

    auto J = [](double t, const arma::rowvec& u) -> arma::mat {
        double x = u(0);
        double y = u(1);
        arma::mat M = {{     0      ,   1   },
                       { -2*mu*x*y - 1 , mu*(1-x*x) }};
        return M;
    };

    arma::vec t = {t0, tf};
    arma::mat U = {U0_1, U0_2};

    rk45 RK45;
    RK45.adaptive_max_err = 1e-6;

    auto evnt = [](double t, const arma::rowvec& U) -> double {
        return U(0) - (-1.6); // when u = -1.6
    };

    if (add_event) RK45.add_stopping_event(evnt, event_direction::POSITIVE); // enable event
    RK45.ode_solve(f,t,U); // we will use our rk45() approximation as the exact solution

    ddvec tt = arma::conv_to<ddvec>::from(t);
    ddvec uu0 = arma::conv_to<ddvec>::from(U.col(0));
    // ddvec uu1 = arma::conv_to<ddvec>::from(U.col(1));
    matplotlibcpp::named_plot("U1 - exact", tt, uu0, "-r");
    // matplotlibcpp::named_plot("U2 - exact", tt, uu1, "-b");

    t = {t0, tf};
    U = {U0_1, U0_2};

    // test ode solvers here
    // am1 dsolver; std::cout << "using Adam's Multon O(k) method, i.e. implicit Euler..." << std::endl;
    // am2 dsolver; std::cout << "using Adam's Multon O(k^2) method, i.e. trapezoid rule..." << std::endl;
    // rk4 dsolver; std::cout << "using Runge Kutta O(k^4) method..." << std::endl;
    // rk5i dsolver; std::cout << "using diagonally implicit Runge Kutta O(k^5) method..." << std::endl;
    rk45i dsolver; std::cout << "using adaptive diagonally implicit Runge Kutta O(k^4-->5) method..." << std::endl;
    
    // dsolver.step = 1/(5*mu);
    dsolver.adaptive_max_err = 1e-4;
    if (add_event) dsolver.add_stopping_event(evnt, event_direction::POSITIVE);
    dsolver.ode_solve(f,t,U);
    // dsolver.ode_solve(f,J,t,U); // am1, am2, rk5i, rk45i

    tt = arma::conv_to<ddvec>::from(t);
    uu0 = arma::conv_to<ddvec>::from(U.col(0));
    // uu1 = arma::conv_to<ddvec>::from(U.col(1));
    matplotlibcpp::named_plot("U1 - test", tt, uu0, "*r");
    // matplotlibcpp::named_plot("U2 - test", tt, uu1, "*b");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}