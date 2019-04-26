#include "numerics.hpp"

using namespace numerics;

// g++ -g -Wall -o minimize minimize_ex.cpp -lnumerics -larmadillo

typedef std::vector<double> vvd;

void wait_for_key();

int main() {
    int n = 2; // n=2 for rosenbrock, anything for the others
    arma::arma_rng::set_seed_random();

    arma::vec x;
    // x = 5 - 10*arma::randu(n); // styb-tang or sphere
    x = {4*arma::randu()-2, 4*arma::randu() - 1}; // rosenbrock
    if (n <= 5) std::cout << "initial point: " << x.t() << std::endl;

    arma::vec w = arma::linspace(-5,5,n); // for sphere
    
    arma::vec tru_min;
    tru_min = arma::ones(n);
    // tru_min *= -2.9035; // for styb-tang

    // sphere function
    // styblinski tang
    // rosenbrock
    vec_dfunc g = [&w](const arma::vec& x) -> double {
        // return arma::norm(x - w);
        // return arma::sum(x%x%x%x - 16*x%x + 5*x)/2;
        return std::pow(1-x(0),2) + 100*std::pow(x(1) - x(0)*x(0),2);
    };
    vector_func dg = [&w](const arma::vec& x) -> arma::vec {
        // return 2*(x-w);
        // return (2*x%x%x - 16*x + 2.5);
        return {-2*(1-x(0)) - 400*x(0)*(x(1)-x(0)*x(0)),
                200*(x(1)-x(0)*x(0))};
    };
    sp_vector_func dgi = [&w](const arma::vec& x,int i) -> double {
        // return 2*(x(i) - w(i));
        return (2*x(i)*x(i)*x(i) - 16*x(i) + 2.5);
        // none for rosenbrock
    };
    vec_mat_func J = [](const arma::vec& x) -> arma::mat {
        // return 2*arma::eye(x.n_elem, x.n_elem);
        // return arma::diagmat(6*x%x - 16);
        return {{1200*x(0) - 400*x(1) + 2, -400*x(0)},
                {-400*x(0), 200}};
    };

    optim_opts opts;
    // opts.use_bfgs(&dg); std::cout << "using BFGS..." << std::endl;
    // opts.use_momentum(&dg); std::cout << "using momentum gradient descent..." << std::endl;
    // opts.use_sgd(&dgi); opts.stochastic_batch_size = 5; std::cout << "using batch stochastic gradient descent..." << std::endl;
    // opts.use_lmlsqr(&dg); opts.hessian_func = &J; std::cout << "using Levenberg-Marquardt least squares..." << std::endl;
    // opts.use_lbfgs(&dg); opts.num_iters_to_remember = 5; std::cout << "using limited memory BFGS..." << std::endl;
    // opts.use_nlcgd(&dg); std::cout << "using conjugate gradient method..." << std::endl;
    opts.use_adj_gd(&dg); std::cout << "using adjusted gradient descent..." << std::endl;
    // opts.use_newton(&dg,&J); std::cout << "using Newton's method..." << std::endl;
    opts.tolerance = 1e-5;
    opts.max_iter = 1000;

    clock_t tt = clock();
    double gx = numerics::minimize_unc(g, x, opts);
    tt = clock() - tt;

    arma::vec centers;
    // centers = {-2.903,0.1567,2.746};  // styb-tang
    centers = {-2,-1,0,1,2}; // rosenbrock and sphere
    arma::uvec bins = arma::hist(x,centers);
    int i = arma::index_max(bins);

    std::cout << "\noptimization results:\t\t" << gx << std::endl
              << "true min:\t\t\t" << g(tru_min) << std::endl
              << "mode(x):\t\t\t " << centers(i) << std::endl
              << "minimize_unc() took " << (float)tt/CLOCKS_PER_SEC << " seconds" << std::endl
              << "num iterations needed: " << opts.num_iters_returned << std::endl;
    
    return 0;
}