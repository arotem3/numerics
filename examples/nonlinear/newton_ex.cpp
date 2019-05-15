#include "numerics.hpp"

// g++ -Wall -g -o newton newton_ex.cpp -lnumerics -larmadillo

using namespace numerics;

int main() {
    arma::arma_rng::set_seed_random();

    vector_func f = [](const arma::vec& x) -> arma::vec { // function
        arma::vec y = {0,0,0};
        double a = x(0), b = x(1), c = x(2);
        y(0) = 4*a*a*a - b*b*c;
        y(1) = -2*a*b*c+b;
        y(2) = 2*c - a*b*b;
        return y;
    };

    std::function<arma::mat(const arma::vec&)> J = [](const arma::vec& v) -> arma::mat { // Jacobian
        double x = v(0);
        double y = v(1);
        double z = v(2);
        arma::mat J = {{ 12*x*x, -2*y*z, -y*y},
                       {-2*y*z,  -2*x*z+1, -2*x*y},
                       {-y*y,    -2*x*y,  2}};
        return J;
    };

    std::cout << "In this file you can test the nonlinear solvers: Newton's method, Broyden's method, and Levenberg-Marquardt." << std::endl
              << "we will try to find roots of f(x,y,z) = [4x^3 - z*y^2; -2xyz+y; 2z-x*y^2]" << std::endl
              << "We will use a random initial guess on the interval [-2,2]." << std::endl
              << "we also know there are 5 possible roots:" << std::endl
              << "\t[0,0,0]\n\t[-0.707,-1.414,-0.707]\n\t[-0.707,1.414,-0.707]\n\t[0.707,-1.414,0.707]\n\t[0.707,1.414,0.707]" << std::endl;
    arma::vec x0 = 4*arma::randu(3) - 2;

    nonlin_opts opts;
    opts.use_FD_jacobian = true;
    opts.use_cgd = true;
    opts.jacobian_func = &J;

    // lsqr_opts opts;
    // opts.jacobian_func = &J;
    // opts.use_cgd = true;

    // fpi_opts opts;
    // vector_func g = [&](const arma::vec& x) -> arma::vec {
    //     return f(x) + x;
    // };

    clock_t t = clock();
    // newton(f,J,x0,opts); std::cout << "using Newton's method..." << std::endl;
    broyd(f,x0,opts); std::cout << std::endl << "using Broyden's method..." << std::endl;
    // lmlsqr(f,x0,opts); std::cout << "using Levenberg-Marquardt least squares..." << std::endl;
    // mix_fpi(g,x0,opts); std::cout << "using fixed point iteration ..." << std::endl;
    t = clock() - t;

    arma::vec F = f(x0);

    std::cout << "results:" << std::endl << std::fixed << std::setprecision(4)
              << "\troot:       [" << x0(0) << ",   " << x0(1) << ",   " << x0(2) << "]" << std::endl
              << "\tf(root):    [" << F(0) << ",   " << F(1) << ",   " << F(2) << "]" << std::endl
              << "\ttime: " << (float)t/CLOCKS_PER_SEC << " secs" << std::endl
              << "\ttotal iterations needed: " << opts.num_iters_returned << std::endl << std::endl;

    // std::cout << "Approximation of jacobian at function exit:" << std::endl << opts.final_jacobian << std::endl;
    // std::cout << "actual jacobian:" << std::endl << J(x0) << std::endl;
    
    return 0;
}