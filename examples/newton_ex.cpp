#include "../numerics.hpp"
#include <iomanip>

// g++ -Wall -g -o newton_ex examples/newton_ex.cpp newton.cpp lmlsqr.cpp finite_dif.cpp -larmadillo

using namespace numerics;

int main() {
    auto f = [](const arma::vec& x) -> arma::vec { // function
        arma::vec y = {0,0,0};
        y(0) = x(0)*x(1) - 1;
        y(1) = x(0) + x(1)*x(2);
        y(2) = 2*x(1) - x(2);
        return y;
    };

    std::function<arma::mat(const arma::vec&)> J = [](const arma::vec& v) -> arma::mat { // Jacobian
        double x = v(0);
        double y = v(1);
        double z = v(2);
        arma::mat J = {{y,x,0},
                       {1,z,y},
                       {0,2,-1}};
        return J;
    };

    arma::vec root = {-1.25992, -0.793701, -1.5874};

    std::cout << "In this file you can test the nonlinear solvers: Newton's method, BFGS, Broyden's method, and Levenberg-Marquardt." << std::endl;
    std::cout << "we will try to find roots of f(x,y,z) = [xy - 1, x + yz, 2y - z] with initial guess [-1,-1,-1]." << std::endl;
    std::cout << "for newton we will need the jacobian: " << std::endl << "[y   x   0]\n[1   z   y]\n[0   2  -1]" << std::endl;
    std::cout << "we also know there is only one root at: [-1.25992, -0.793701, -1.5874]" << std::endl;
    arma::vec x0 = {-1,-1,-1};

    nonlin_opts opts;
    opts.use_FD_jacobian = true;
    // lsqr_opts opts;
    // opts.jacobian_func = &J;

    clock_t t = clock();
    // bfgs(f,x0,opts);
    // newton(f,J,x0,opts);
    broyd(f,x0,opts);
    // lmlsqr(f,x0,opts);
    t = clock() - t;

    arma::vec F = f(x0);
    arma::vec error = root - x0;

    std::cout << "newton(f,J,x0):" << std::endl << std::fixed << std::setprecision(4) << std::fixed
              << "\troot:       [" << x0(0) << ",   " << x0(1) << ",   " << x0(2) << "]" << std::endl
              << "\tf(root):    [" << F(0) << ",   " << F(1) << ",   " << F(2) << "]" << std::endl
              << "\terror:      [" << error(0) << ",   " << error(1) << ",   " << error(2) << "]" << std::endl
              << "\ttime: " << (float)t/CLOCKS_PER_SEC << " secs" << std::endl
              << "\ttotal iterations needed: " << opts.num_iters_returned << std::endl << std::endl;

    std::cout << "Approximation of jacobian at function exit:" << std::endl << opts.final_jacobian << std::endl;
    std::cout << "actual jacobian:" << std::endl << J(x0) << std::endl;
    
    return 0;
}