#include "numerics.hpp"

// g++ -Wall -g -o newton newton_ex.cpp -O3 -lnumerics -larmadillo

const std::string methods[] = {"newton","broyden","lm","trust"};
bool in_methods(const std::string& s) {
    return (std::count(methods, methods+4, s) > 0);
}

arma::vec f(const arma::vec& v) {
    arma::vec fv = {0,0,0};
    double x = v(0), y = v(1), z = v(2);
    fv(0) = 4*std::pow(x,3) - std::pow(y,2)*z;
    fv(1) = -2*z*y*z + y;
    fv(2) = 2*z - x*std::pow(y,2);
    return fv;
}

arma::mat J(const arma::vec& v) {
    double x = v(0);
    double y = v(1);
    double z = v(2);
    arma::mat Jac = {{ 12*x*x, -2*y*z, -y*y},
                    {-2*y*z,  -2*x*z+1, -2*x*y},
                    {-y*y,    -2*x*y,  2}};
    return Jac;
}

int main() {
    std::cout << "In this file you can test the nonlinear solvers: Newton's method, Broyden's method, and Levenberg-Marquardt." << std::endl
              << "Consider, f(x,y,z) = [4x^3 - z*y^2; -2xyz+y; 2z-x*y^2]" << std::endl
              << "We will use a random initial guess." << std::endl
              << "The solvers are:" << std::endl
              << "\t'newton' : Newton's method with line search." << std::endl
              << "\t'broyden' : Inexact Newton's method with line search using Broyden's update for the jacobian." << std::endl
              << "\t'lm' : Levenberg-Marquardt algorithm for nonlinear least-squares." << std::endl
              << "\t'trust' : Trust-region method constrained to 2D subspace minimization." << std::endl
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
    
    arma::arma_rng::set_seed_random();
    arma::vec x = arma::randn(3);

    numerics::optimization::QausiNewton *fsolver;

    if (choice == "newton") fsolver = new numerics::optimization::Newton();
    else if (choice == "broyden") fsolver = new numerics::optimization::Broyden();
    else if (choice == "lm") fsolver = new numerics::optimization::LmLSQR();
    else fsolver = new numerics::optimization::TrustNewton();

    clock_t t = clock();
    // fsolver->fsolve(x,f,J);
    fsolver->fsolve(x, f); // broyd and lmlsqr do not need a jacobian function
    t = clock() - t;

    arma::vec F = f(x);
    std::string flag = fsolver->get_exit_flag();

    std::cout << "results:" << std::endl << std::fixed << std::setprecision(4)
              << "\troot:       [" << x(0) << ",   " << x(1) << ",   " << x(2) << "]" << std::endl
              << "\tf(root):    [" << F(0) << ",   " << F(1) << ",   " << F(2) << "]" << std::endl
              << "\ttime: " << (float)t/CLOCKS_PER_SEC << " secs" << std::endl
              << "\ttotal iterations needed: " << fsolver->n_iter << std::endl
              << "\texit flag: " << flag << std::endl;
    
    return 0;
}