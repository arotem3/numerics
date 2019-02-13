#include "ODE.hpp"
#include "plot.hpp"

// g++ -g -Wall -o diffeq examples/ode_ex.cpp examples/wait.cpp -lnumerics -larmadillo

using namespace numerics;
using namespace ODE;

void wait_for_key();

const double t0 = 0;
const double tf = 20;
const double U0_1 = 3;
const double U0_2 = 0;

int main() {
    std::cout << "Let's solve the vanderpol equation with mu=1." << std::endl
              << "We will assume the true solution is the output of rk45 with error < 1e-6." << std::endl
              << "you can play around with the solvers and their options within the example code." << std::endl << std::endl;

    odefun f = [](double t, const arma::rowvec& u) -> arma::rowvec {
        arma::rowvec v(2);
        double x = u(0);
        double y = u(1);
        v(0) = y;
        v(1) = (1-x*x)*y - x; // vanderpol with mu = 1
        return v;
    };

    odeJac J = [](double t, const arma::rowvec& u) -> arma::mat {
        double x = u(0);
        double y = u(1);
        arma::mat M = {{     0      ,   1   },
                       { -2*x*y - 1 , 1-x*x }};
        return M;
    };

    arma::vec t = {t0, tf};
    arma::mat U = {U0_1, U0_2};

    ivp_options opts;
    opts.adaptive_max_err = 1e-6;
    opts.adaptive_step_min = rk45_kmin;
    opts.adaptive_step_max = rk45_kmax;
    opts.ode_jacobian = &J;

    event_func evnt = [](double t, const arma::rowvec& U) -> event_out {
        event_out event;
        event.val = U(0) - (-1.6); // when u = -1.6
        event.dir = event_direction::POSITIVE; // when the quantity [U(0) - (-1.6)] goes from negative --> positive.
        return event;
    };

    // opts.events.push_back(evnt); // enable events
    rk45(f,t,U,opts); // we will use our rk45() approximation as the exact solution

    Gnuplot fig;

    lines(fig, t, (arma::mat)U.col(0), {{"legend","U1 - exact"}});
    lines(fig, t, (arma::mat)U.col(1), {{"legend","U2 - exact"}});

    t = {t0, tf};
    U = {U0_1, U0_2};

    opts.standard_adaptive();
    opts.step = 0.2;

    // test ode solvers here
    // am1(f,t,U,opts); std::cout << "using Adam's Multon O(k) method, i.e. implicit Euler..." << std::endl;
    // am2(f,t,U,opts); std::cout << "using Adam's Multon O(k^2) method, i.e. trapezoid rule..." << std::endl;
    // rk4(f,t,U, opts); std::cout << "using Runge Kutta O(k^4) method..." << std::endl;
    rk5i(f,t,U, opts); std::cout << "using diagonally implicit Runge Kutta O(k^5) method..." << std::endl;
    // bdf23(f,t,U,opts); std::cout << "using adaptive backwards differentiation formula O(k^2)..." << std::endl;
    /* dsolnc soln = ivp(f, t, U, opts, BDF23); std::cout << "using general solver and interpolating..." << std::endl;
        t = arma::linspace(t0,t(t.n_elem-1),1000);
        U = soln(t); */

    plot(fig, t, (arma::mat)U.col(0), {{"legend","U1 - test"},{"linespec","or"},{"markersize","1.5"}});
    plot(fig, t, (arma::mat)U.col(1), {{"legend","U2 - test"},{"linespec","ob"},{"markersize","1.5"}});

    wait_for_key();

    return 0;
}