#include "numerics.hpp"

using namespace numerics;

// g++ -g -Wall -o minimize minimize_ex.cpp -O3 -lnumerics -larmadillo

typedef std::vector<double> vvd;

void wait_for_key();

int main() {
    std::cout << "objective function:" << std::endl
              << "\t(1) rosenbrock" << std::endl
              << "\t(2) quadratic" << std::endl
              << "\t(3) stylinsky-tang" << std::endl
              << "your choice : ";
    int choice;
    std::cin >> choice;

    int n;
    if (choice == 1) n = 2;
    else n = 4;
    arma::arma_rng::set_seed_random();

    arma::vec x;
    if (choice == 1) x = arma::randn(2);
    else x = arma::randn(n);
    if (n <= 5) std::cout << "initial point: " << x.t() << std::endl;

    arma::vec b = arma::randn(n);
    arma::mat A = arma::randn(2*n,n); A = A.t() * A;

    
    arma::vec tru_min;
    if (choice == 1) tru_min = {1, 1};
    else if (choice == 2) tru_min = arma::solve(A,b);
    else tru_min = -2.9035*arma::ones(n);

    auto g = [&A,&b,choice](const arma::vec& x) -> double {
        if (choice == 1) {
            return std::pow(1-x(0),2) + 100*std::pow(x(1) - x(0)*x(0),2);
        } else if (choice == 2) {
            arma::vec w = A*x - b;
            return 0.5*arma::dot(w,w);
        } else return arma::sum(arma::pow(x,4) - 16*arma::pow(x,2) + 5*x)/2;
    };
    auto dg = [&A,&b,choice](const arma::vec& x) -> arma::vec {
        if (choice == 1) {
            return {-2*(1-x(0)) - 400*x(0)*(x(1)-x(0)*x(0)),
                    200*(x(1)-x(0)*x(0))};
        } else if (choice == 2) {
            return A*x - b;
        } else return (2*arma::pow(x,3) - 16*x + 2.5);
    };
    auto H = [&A,&b,choice](const arma::vec& x) -> arma::mat {
        if (choice == 1) {
            return {{1200*x(0) - 400*x(1) + 2, -400*x(0)},
                    {-400*x(0), 200}};
        } else if (choice == 2) {
            return A;
        } else return arma::diagmat(6*x%x - 16);
    };

    std::cout << "optimization method:\n"
              << "\t(1) nelder_mead\n\t\trequires only objective function. also known as simplex method.\n"
              << "\t(2) bfgs\n\t\tquasi-newton, requires gradient info\n"
              << "\t(3) lbfgs\n\t\tquasi-newton, requires gradient info. more efficient that BFGS per step, may require more function step.\n"
              << "\t(4) mgd\n\t\tmomentum gradient descent, requires gradient info. more efficient than quasi-newton per step, requires many steps.\n"
              << "\t(5) nlgcd\n\t\tnon-linear conjugate gradient descent, requires gradient info. requires fewer function evaluations than most methods.\n"
              << "\t(6) adj_gd\n\t\tadjusted gradient descent, requires gradient info. More function evals than nlgdc.\n";
    std::cin >> choice;
    clock_t tt = clock();
    if (choice < 1 || choice > 6) {
        choice = 1;
        std::cout << "invalid choice for optimization.\n";
    }
    std::string flag;
    int n_iter=0;
    if (choice == 1) {
        std::cout << "using Nelder-Mead method..." << std::endl;
        numerics::nelder_mead fmin;
        // fmin.tol = ;
        // fmin.max_iterations = ;
        fmin.minimize(g,x);
        fmin.get_exit_flag(flag);
        n_iter = fmin.num_iterations();
    } else if (choice == 2) {
        std::cout << "using BFGS..." << std::endl;
        numerics::bfgs fmin;
        // fmin.tol = ;
        // fmin.max_iterations = ;
        fmin.minimize(g, dg, x);
        // fmin.minimize(g, dg, H, x); // use Hessian information to improve results
        fmin.get_exit_flag(flag);
        n_iter = fmin.num_iterations();
    } else if (choice == 3) {
        std::cout << "using limited memory BFGS..." << std::endl;
        numerics::lbfgs fmin;
        // fmin.tol = ;
        // fmin.max_iterations = ;
        fmin.minimize(g, dg, x);
        // fmin.minimize(g, dg, H, x); // use Hessian information to improve results
        fmin.get_exit_flag(flag);
        n_iter = fmin.num_iterations();
    } else if (choice == 4) {
        std::cout << "using momentum gradient descent..." << std::endl;
        numerics::mgd fmin;
        // fmin.step_size = ;
        // fmin.tol = ;
        // fmin.max_iterations = ;
        fmin.minimize(dg,x);
        fmin.get_exit_flag(flag);
        n_iter = fmin.num_iterations();
    } else if (choice == 5) {
        std::cout << "using conjugate gradient method..." << std::endl;
        numerics::nlcgd fmin;
        // fmin.step_size = ;
        // fmin.tol = ;
        // fmin.max_iterations = ;
        fmin.minimize(dg,x);
        fmin.get_exit_flag(flag);
        n_iter = fmin.num_iterations();
    } else { // choice == 6
        std::cout << "using adjusted gradient descent..." << std::endl;
        numerics::adj_gd fmin;
        // fmin.step_size = ;
        // fmin.tol = ;
        // fmin.max_iterations = ;
        fmin.minimize(dg,x);
        fmin.get_exit_flag(flag);
        n_iter = fmin.num_iterations();
    }
    tt = clock() - tt;    

    std::cout << "\noptimization results:\t\t" << g(x) << std::endl
              << "true min:\t\t\t" << g(tru_min) << std::endl;
    if (n <= 5) std::cout << "final x: " << x.t() << "true argmin: " << tru_min.t();
    std::cout << "minimize_unc() took " << (float)tt/CLOCKS_PER_SEC << " seconds" << std::endl
              << "num iterations needed: " << n_iter << std::endl
              << "exit flag: " << flag << std::endl;
    return 0;
}