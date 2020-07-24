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
              << "\t(1) NelderMead\n\t\trequires only objective function. also known as simplex method.\n"
              << "\t(2) BFGS\n\t\tquasi-newton, requires gradient info\n"
              << "\t(3) LBFGS\n\t\tquasi-newton, requires gradient info. more efficient that BFGS per step, may require more function step.\n"
              << "\t(4) MomentumGD\n\t\tmomentum gradient descent, requires gradient info. more efficient than quasi-newton per step, requires many steps.\n";
    do {
        std::cin >> choice;
        if (choice < 1 || choice > 4) {
            std::cout << "invalid choice for optimization. Please specify another option: ";
        }
    } while (choice < 1 || choice > 4);
    
    std::string flag;
    int n_iter=0;
    clock_t tt = clock();
    if (choice == 1) {
        std::cout << "using Nelder-Mead method..." << std::endl;
        numerics::optimization::NelderMead fmin(1e-8,1000);
        fmin.minimize(x,g);
        flag = fmin.get_exit_flag();
        n_iter = fmin.n_iter;
    } else if (choice == 2) {
        std::cout << "using BFGS..." << std::endl;
        numerics::optimization::BFGS fmin(1e-8, 200);
        // fmin.minimize(x, g, dg);
        fmin.minimize(x, g, dg, H); // use Hessian information to improve results
        flag = fmin.get_exit_flag();
        n_iter = fmin.n_iter;
    } else if (choice == 3) {
        std::cout << "using limited memory BFGS..." << std::endl;
        numerics::optimization::LBFGS fmin(5, 1e-8, 200);
        fmin.minimize(x, g, dg);
        flag = fmin.get_exit_flag();
        n_iter = fmin.n_iter;
    } else if (choice == 4) {
        std::cout << "using momentum gradient descent..." << std::endl;
        numerics::optimization::MomentumGD fmin(1e-5, 1000);
        // fmin.set_step_size(0.001, 0.9);
        fmin.minimize(x,g,dg);
        flag = fmin.get_exit_flag();
        n_iter = fmin.n_iter;
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