#include "../ODEs/ODE.hpp"
#include "gnuplot_i.hpp"

using namespace numerics;
using namespace ODE;

void wait_for_key();

const double t0 = 0;
const double tf = 10;
const double U0_1 = 0;
const double U0_2 = 3;

int main() {
    auto f = [](double t, const arma::rowvec& u) -> arma::rowvec {
        arma::rowvec v(2);
        double x = u(0);
        double y = u(1);
        v(0) = std::exp(-t)*y;
        v(1) = std::sin(x);
        return v;
    };
    arma::vec t = {t0, tf};
    arma::mat U = {U0_1, U0_2};

    ivp_options opts;
    opts.adaptive_max_err = 1e-8;
    opts.adaptive_step_min = rk45_kmin;
    opts.adaptive_step_max = rk45_kmax;
    /* opts.events.push_back( [](double t, const arma::rowvec& U) -> event_out {
        event_out event;
        event.val = U(0) - 3;
        event.dir = ALL;
        return event;
    }  ); // enable events */
    rk45(f,t,U,opts); // we will use our rk45() approximation as the exact solution

    Gnuplot fig1("test");
    fig1.set_style("lines");

    typedef std::vector<double> stdv;
    stdv t1 = arma::conv_to<stdv>::from(t);
    stdv U1 = arma::conv_to<stdv>::from(U.col(0));
    stdv U2 = arma::conv_to<stdv>::from(U.col(1));

    fig1.plot_xy(t1,U1,"U1 - exact");
    fig1.plot_xy(t1,U2,"U2 - exact");

    fig1.set_style("points");
    t = {t0, tf};
    U = {U0_1, U0_2};

    opts.standard_adaptive();
    opts.step = 0.1;

    // test ode solvers here
    // am1(f,t,U,opts);
    am2(f,t,U,opts);
    // rk4(f,t,U, opts);
    // rk5i(f,t,U, opts);
    // bdf23(f,t,U,opts);
    /* dsolnc soln = IVP_solve(f, t, U, opts, BDF23);
        t = arma::linspace(t0,t(t.n_elem-1),1000);
        U = soln(t); */

    t1 = arma::conv_to<stdv>::from(t);
    U1 = arma::conv_to<stdv>::from(U.col(0));
    U2 = arma::conv_to<stdv>::from(U.col(1));

    fig1.plot_xy(t1,U1,"U1 - test");
    fig1.plot_xy(t1,U2,"U2 - test");

    wait_for_key();

    return 0;
}