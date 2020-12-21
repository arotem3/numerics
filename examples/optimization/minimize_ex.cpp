#include "numerics.hpp"

using namespace numerics;

// g++ -g -Wall -o minimize minimize_ex.cpp -O3 -lnumerics -larmadillo

typedef std::vector<double> vvd;

double g(const arma::vec& x) {
    return std::pow(1-x(0),2) + 100*std::pow(x(1) - x(0)*x(0),2);
}

arma::vec dg(const arma::vec& x) {
    return {-2*(1-x(0)) - 400*x(0)*(x(1)-x(0)*x(0)),
                    200*(x(1)-x(0)*x(0))                };
}

arma::mat H(const arma::vec& x) {
    return {
            {1200*x(0)*x(0) - 400*x(1) + 2, -400*x(0)},
            {               -400*x(0),       200}
        };
}

const std::string methods[] = {"newton","trust","bfgs","lbfgs","gd"};
bool in_methods(const std::string& s) {
    return (std::count(methods, methods+5, s) > 0);
}

int main() {
    std::string choice;
    arma::arma_rng::set_seed_random();
    
    std::cout << "we will minimize the Rosebrock function" << std::endl;
    
    arma::vec x = {-2.0,-1.0};
    std::cout << "initial point: " << x.t() << std::endl;
    
    arma::vec tru_min = {1.0, 1.0};

    std::cout << "The available methods are\n"
              << "\t'newton'   : Newton's method using either exact Hessian inverse or PCG for computing a search direction while also performing a line search.\n\n"
              << "\t'trust'    : Trust-Region method constrained to the 2D subspace spanned by gradient direction and newton direction.\n\n"
              << "\t'bfgs'     : quasi-newton method which constructs an approximation to the Hessian inverse. This method performs a line search every iteration.\n\n"
              << "\t'lbfgs'    : quasi-newton method which approximates the Hessian inverse, but is more efficient than BFGS per step, may require more steps.\n\n"
              << "\t'gd'       : Momentum gradient descent. This method performs either an exact line minimization step or a constant step size with momentum. More efficient than quasi-newton methods per step, requires many steps.\n\n"
              << "solver: ";
    do {
        std::cin >> choice;
        if (in_methods(choice)) break;
        else {
            std::cout << "solver must be one of {";
            for (std::string m : methods) std::cout << m << ",";
            std::cout << "}, try again.\nsolver: ";
        }
    } while (true);
    
    numerics::optimization::GradientOptimizer *solver;

    if (choice == "newton") solver = new numerics::optimization::NewtonMin();
    else if (choice == "trust") solver = new numerics::optimization::TrustMin();
    else if (choice == "bfgs") solver = new numerics::optimization::BFGS();
    else if (choice == "lbfgs") solver = new numerics::optimization::LBFGS();
    else solver = new numerics::optimization::MomentumGD(1e-3, 1e-6);

    clock_t tt = clock();
    solver->minimize(x, g, dg);
    // solver->minimize(x, g, dg, H);
    tt = clock() - tt;

    std::string flag = solver->get_exit_flag();
    int n_iter = solver->n_iter;

    std::cout << "\noptimization results:\t\t" << g(x) << std::endl
              << "true min:\t\t\t" << g(tru_min) << std::endl
              << "final x: " << x.t() << "true argmin: " << tru_min.t()
              << "minimize() took " << (float)tt/CLOCKS_PER_SEC << " seconds" << std::endl
              << "num iterations needed: " << n_iter << std::endl
              << "exit flag: " << flag << std::endl;
    return 0;
}