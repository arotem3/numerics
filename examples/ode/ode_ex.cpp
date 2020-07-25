#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o diffeq ode_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

using namespace numerics::ode;
typedef std::vector<double> dvec;

const double mu = 100; // bigger value --> more stiff
const double t0 = 0;
const double tf = std::max(20.0,2*(3-2*std::log(2))*mu);
const arma::vec& U0 = {2,0};
const bool add_event = false;

int main() {
    std::cout << "Let's solve the vanderpol equation with mu=" << mu << std::endl
              << "We will assume the true solution is the output of rk45 with error < 1e-6." << std::endl
              << "you can play around with the solvers and their options within the example code." << std::endl << std::endl;

    auto f = [](double t, const arma::vec& u) -> arma::vec {
        arma::vec v(2);
        double x = u(0);
        double y = u(1);
        v(0) = y;
        v(1) = mu*(1-x*x)*y - x; // vanderpol with mu = 1
        return v;
    };

    auto J = [](double t, const arma::vec& u) -> arma::mat {
        double x = u(0);
        double y = u(1);
        arma::mat M = {{0,1},
                       {-2*mu*x*y - 1, mu*(1-x*x)}};
        return M;
    };

    rk45 RK45(1e-6);

    auto evnt = [](double t, const arma::vec& U) -> double {
        return U(0) - (-1.6); // when u = -1.6
    };

    if (add_event) RK45.add_stopping_event(evnt, event_direction::POSITIVE); // enable event
    ODESolution sol = RK45.ode_solve(f, t0, tf, U0); // we will use our rk45() approximation as the exact solution

    dvec tt = arma::conv_to<dvec>::from(sol.t);
    dvec uu0 = arma::conv_to<dvec>::from(sol.solution.col(0));
    dvec uu1 = arma::conv_to<dvec>::from(sol.solution.col(1));
    matplotlibcpp::subplot(1,2,1);
    matplotlibcpp::named_plot("U1 - exact", tt, uu0, "-r");
    matplotlibcpp::subplot(1,2,2);
    matplotlibcpp::named_plot("U2 - exact", tt, uu1, "-b");

    // test ode solvers here
    // am1 dsolver(1/(10*mu)); std::cout << "using Adam's Multon O(k) method, i.e. implicit Euler...\n";
    // am2 dsolver(1/(5*mu)); std::cout << "using Adam's Multon O(k^2) method, i.e. trapezoid rule...\n";
    // rk4 dsolver(1/(5*mu)); std::cout << "using Runge Kutta O(k^4) method...\n";
    // rk5i dsolver(1/(mu)); std::cout << "using diagonally implicit Runge Kutta O(k^5) method...\n";
    rk45i dsolver; std::cout << "using adaptive diagonally implicit Runge Kutta O(k^4->5) method...\n";
    
    if (add_event) dsolver.add_stopping_event(evnt, event_direction::POSITIVE);
    ODESolution sol1 = dsolver.ode_solve(f,t0,tf,U0);
    // ODESolution sol1 = dsolver.ode_solve(f,J,t0,tf,U0);

    tt = arma::conv_to<dvec>::from(sol1.t);
    uu0 = arma::conv_to<dvec>::from(sol1.solution.col(0));
    uu1 = arma::conv_to<dvec>::from(sol1.solution.col(1));
    matplotlibcpp::subplot(1,2,1);
    matplotlibcpp::named_plot("U1 - test", tt, uu0, "*k");
    matplotlibcpp::subplot(1,2,2);
    matplotlibcpp::named_plot("U2 - test", tt, uu1, "*k");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}