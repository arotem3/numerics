#include "../ODEs/ODE.hpp"
#include "gnuplot_i.hpp"

double afunc(double x) {
    return std::sin(x);
}

void wait_for_key();

typedef std::vector<double> stdv;

using namespace ODE;

int main() {
    std::cout << "We will solve the linear boundary value problem:" << std::endl
              << "\tu''(x) = sin(x) + u(x)\t0 < x < 2*pi" << std::endl
              << "\tu(0) = 1\tu(2*pi) + u'(2*pi) = 0" << std::endl;
    linear_BVP problem;
    problem.set_boundaries(0,2*M_PI); // 0 < x < 2*pi
    problem.set_LBC(1,0,1); // u(0) = 1
    problem.set_RBC(1,1,0); // u(2pi) + u'(2pi) = 0
    // u''(x) = sin(x) + u(x)
    problem.set_a( afunc );
    problem.set_b(1.0);
    
    arma::vec x;
    arma::mat U;

    problem.solve(x,U,40); // 4th order FD approximation
    // problem.solve(x,U,40, SECOND_ORDER); // 2nd order FD approximation
    // problem.solve(x,U,20, CHEBYSHEV); // spectral order approximation
    // dsolnp y = problem.solve(20); x = arma::linspace(0,2*M_PI); U = y.soln(x); // spectral order approximation outputing cheb polynomial

    arma::mat u = 0.25 * (arma::exp(-2*M_PI - x) % (-1 + 4*std::exp(2*M_PI)+arma::exp(2*x)) - 2*arma::sin(x));

    Gnuplot graph;
    stdv x1 = arma::conv_to<stdv>::from(x);
    stdv U1 = arma::conv_to<stdv>::from(U);
    stdv u1 = arma::conv_to<stdv>::from(u);

    graph.set_style("points");
    graph.plot_xy(x1,U1,"U - solution");
    graph.set_style("lines");
    graph.plot_xy(x1,u1,"u - exact");

    wait_for_key();

    std::cout << "Now we will solve the nonlinear boundary value problem:" << std::endl
              << "\tu' = v" << std::endl << "\tv' = -sin(u)" << std::endl
              << "\t0 < x < 2*pi" << std::endl
              << "\tu(0) = 1\tu(2*pi) = 1" << std::endl
              << "we will use an initial guess of u(x) = cos(x) and v(x) = sin(x) i.e. the result from linearization" << std::endl
              << "we will also use an intial guess of u(x) = 1 and v(x) = 0 as another example.\n(this will demonstrate non-uniqueness depending on your choice of solver)." << std::endl;

    odefun f = [](double x, const arma::rowvec& u) -> arma::rowvec {
        arma::rowvec up(2, arma::fill::zeros);
        up(0) = u(1);
        up(1) = -std::sin(u(0));
        return up;
    };

    std::function<arma::mat(double,const arma::rowvec&)> J = [](double x,const arma::rowvec& u) -> arma::mat {
        arma::mat A(2,2,arma::fill::zeros);
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
    opts.num_points = 100;
    // opts.jacobian_func = &J; // providing a jacobian function improves runtime significantly

    dsolnp soln = bvp(f, bc, guess, opts);
    x1 = arma::conv_to<stdv>::from(soln.independent_var_values);
    U1 = arma::conv_to<stdv>::from(soln.solution_values.col(0));
    stdv U2 = arma::conv_to<stdv>::from(soln.solution_values.col(1));

    graph.reset_plot();
    graph.set_style("lines");
    graph.plot_xy(x1,U1,"u(x) -- guess of cos(x)");
    graph.plot_xy(x1,U2,"v(x) -- guess of sin(x)");

    std::cout << "Number of nonlinear iterations needed by solver: " << opts.nlnopts.num_iters_returned << std::endl;

    guess = [](const arma::vec& x) -> arma::mat {
        arma::mat y = arma::zeros(x.n_elem,2);
        y.col(0) = arma::ones(x.n_elem);
        y.col(1) = arma::zeros(x.n_elem);
        return y;
    };

    soln = bvp(f, bc, guess, opts);
    x1 = arma::conv_to<stdv>::from(soln.independent_var_values);
    U1 = arma::conv_to<stdv>::from(soln.solution_values.col(0));
    U2 = arma::conv_to<stdv>::from(soln.solution_values.col(1));

    graph.plot_xy(x1,U1,"u(x) -- guess of 1");
    graph.plot_xy(x1,U2,"v(x) -- guess of 0");

    std::cout << "Number of nonlinear iterations needed by solver: " << opts.nlnopts.num_iters_returned << std::endl;

    wait_for_key();

    return 0;
}