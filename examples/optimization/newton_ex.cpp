#include "numerics.hpp"

// g++ -Wall -g -o newton newton_ex.cpp -O3 -lnumerics -larmadillo

using namespace numerics;

int main() {
    arma::arma_rng::set_seed_random();

    auto f = [](const arma::vec& x) -> arma::vec { // function
        arma::vec y = {0,0,0};
        double a = x(0), b = x(1), c = x(2);
        y(0) = 4*a*a*a - b*b*c;
        y(1) = -2*a*b*c+b;
        y(2) = 2*c - a*b*b;
        return y;
    };

    auto J = [](const arma::vec& v) -> arma::mat { // Jacobian
        double x = v(0);
        double y = v(1);
        double z = v(2);
        arma::mat Jac = {{ 12*x*x, -2*y*z, -y*y},
                       {-2*y*z,  -2*x*z+1, -2*x*y},
                       {-y*y,    -2*x*y,  2}};
        return Jac;
    };

    std::cout << "In this file you can test the nonlinear solvers: Newton's method, Broyden's method, and Levenberg-Marquardt." << std::endl
              << "we will try to find roots of f(x,y,z) = [4x^3 - z*y^2; -2xyz+y; 2z-x*y^2]" << std::endl
              << "We will use a random initial guess." << std::endl
              << "we also know there are 5 possible roots:" << std::endl
              << "\t[0,0,0]\n\t[-0.707,-1.414,-0.707]\n\t[-0.707,1.414,-0.707]\n\t[0.707,-1.414,0.707]\n\t[0.707,1.414,0.707]" << std::endl;
    arma::vec x = arma::randn(3);

    // numerics::optimization::Newton fsolver; std::cout << "using Newton's method..." << std::endl;
    numerics::optimization::Broyd fsolver; std::cout << std::endl << "using Broyden's method..." << std::endl;
    // numerics::optimization::LmLSQR fsolver; std::cout << "using Levenberg-Marquardt least squares..." << std::endl;

    clock_t t = clock();
    fsolver.fsolve(x,f,J);
    // fsolver.fsolve(f,x); // broyd and lmlsqr do not need a jacobian function
    t = clock() - t;

    arma::vec F = f(x);
    std::string flag = fsolver.get_exit_flag();

    std::cout << "results:" << std::endl << std::fixed << std::setprecision(4)
              << "\troot:       [" << x(0) << ",   " << x(1) << ",   " << x(2) << "]" << std::endl
              << "\tf(root):    [" << F(0) << ",   " << F(1) << ",   " << F(2) << "]" << std::endl
              << "\ttime: " << (float)t/CLOCKS_PER_SEC << " secs" << std::endl
              << "\ttotal iterations needed: " << fsolver.n_iter << std::endl
              << "\texit flag: " << flag << std::endl;
    
    return 0;
}