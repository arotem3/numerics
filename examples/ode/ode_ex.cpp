#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o diffeq ode_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

using namespace numerics::ode;
typedef std::vector<double> dvec;

const std::string methods[] = {"am2","rk4","rk5i","rk34i","rk45"};
bool in_methods(const std::string& s) {
    return (std::count(methods, methods+5, s) > 0);
}

const double mu = 100; // bigger value --> more stiff

arma::vec f(double t, const arma::vec& u) {
    arma::vec v(2);
    v(0) = u(1);
    v(1) = mu*(1-u(0)*u(0))*u(1) - u(0); // vanderpol with mu = 1
    return v;
}

arma::mat J(double t, const arma::vec& u) {
    double x = u(0);
    double y = u(1);
    arma::mat M = {{            0,          1},
                   {-2*mu*x*y - 1, mu*(1-x*x)}};
    return M;
}

double evnt(double t, const arma::vec& u) {
    return u(0) + 1.6;
}

int main() {
    const double t0 = 0;
    const double tf = std::max(20.0,2*(3-2*std::log(2))*mu);
    const arma::vec& U0 = {2,0};
    const bool add_event = false;

    std::cout << "Let's solve the vanderpol equation with mu=" << mu << std::endl
              << "We will assume the true solution is the output of rk45 with error < 1e-6." << std::endl
              << "The solvers are:" << std::endl
              << "\t'am2' : trapezoid rule" << std::endl
              << "\t'rk4' : fourth order 5-stage explicit Runge-Kutta" << std::endl
              << "\t'rk5i' : diag-implicit fifth order Runge-Kutta" << std::endl
              << "\t'rk34i' : adaptive diag-implicit fourth order Runge-Kutta" << std::endl
              << "\t'rk45' : adaptive fourth order Dormand-Prince method." << std::endl
              << "solver: ";

    std::string choice;
    do {
        std::cin >> choice;
        if (in_methods(choice)) break;
        else {
            std::cout << "solver must be one of {";
            for (std::string m : methods) std::cout << m << ",";
            std::cout << "}, try again.\nsolver: ";
        }
    } while (true);

    ivpOpts opts; opts.atol = 1e-8; opts.rtol = 1e-6;
    rk45 RK45(opts);

    if (add_event) RK45.add_stopping_event(evnt, "inc"); // enable event
    RK45.solve_ivp(f, t0, tf, U0); // we will use our rk45() approximation as the exact solution

    arma::mat U; arma::vec t;
    RK45.as_mat(t,U);

    InitialValueProblem* dsolver;
    ivpOpts opts1;

    if (choice == "am2") {opts1.cstep = 1/(4*mu); dsolver = new am2(opts1);}
    else if (choice == "rk4") {opts1.cstep = 1/(2*mu); dsolver = new rk4(opts1);}
    else if (choice == "rk5i") {opts1.cstep = 1/mu; dsolver = new rk5i(opts1);}
    else if (choice == "rk34i") dsolver = new rk34i(opts1);
    else dsolver = new rk45(opts1);
    
    if (add_event) dsolver->add_stopping_event(evnt, "inc");
    dsolver->solve_ivp(f,t0,tf,U0);
    // dsolver->solve_ivp(f,J,t0,tf,U0);

    arma::mat V; arma::vec s;
    dsolver->as_mat(s,V);

    matplotlibcpp::subplot(1,2,1);
    dvec uu = arma::conv_to<dvec>::from(U.row(0));
    matplotlibcpp::named_plot("U1 - exact", RK45.t, uu, "-r");
    uu = arma::conv_to<dvec>::from(V.row(0));
    matplotlibcpp::named_plot("U1 - test", dsolver->t, uu, "*k");


    matplotlibcpp::subplot(1,2,2);
    uu = arma::conv_to<dvec>::from(U.row(1));
    matplotlibcpp::named_plot("U2 - exact", RK45.t, uu, "-b");
    uu = arma::conv_to<dvec>::from(V.row(1));
    matplotlibcpp::named_plot("U2 - test", dsolver->t, uu, "*k");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}