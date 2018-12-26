#include "../numerics.hpp"
#include "gnuplot_i.hpp"
#include <iomanip>

// g++ -g -Wall -o lmlsqr_ex examples/lmlsqr_ex.cpp lmlsqr.cpp finite_dif.cpp examples/wait.cpp -larmadillo

using namespace numerics;

void wait_for_key();

typedef std::vector<double> stdv;

int main() {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "lmlsqr() uses the Levenberg-Marquardt algorithm to find least squares solutions of root finding problems f(x) = 0" << std::endl;
    std::cout << "We will try to perform a nonlinear fit of the form:" << std::endl
              << "\ty = f(x) = b(0) * sin(b(1) * x) + b(2) * sin(b(3) * x)" << std::endl
              << "we will take the actual parameters to be b = [1, 2, 1.2, pi/2]" << std::endl;
    
    arma::arma_rng::set_seed_random();
    arma::vec x = 2*M_PI*arma::randu(1000);
    arma::vec t = arma::linspace(0,2*M_PI);
    arma::vec y_true = arma::sin(2*t) + 1.2*arma::sin(M_PI_2*t);
    arma::vec y = arma::sin(2*x) + 1.2*arma::sin(M_PI_2*x) + arma::randn(arma::size(x));

    arma::vec b = {1, 2, 1.2, M_PI_2};
    arma::vec b_hat = b + 0.5*arma::randn(4);
    std::cout << "we will use initial guess b_hat = ["
              << b_hat(0) << ", " << b_hat(1) << ", " << b_hat(2) << ", " << b_hat(3)
              << "] because it is close to our actual parameters" << std::endl << std::endl;

    vector_func f = [x,y](const arma::vec& b) -> arma::vec {
        return b(0)*arma::sin(b(1)*x) + b(2)*arma::sin(b(3)*x) - y;
    };

    // providing a jacobian function provides a substantial performance boost, you can try it with or without it.
    vec_mat_func J = [x](const arma::vec& b) -> arma::mat {
        arma::mat A(x.n_elem,4,arma::fill::zeros);
        A.col(0) = arma::sin(b(1) * x);
        A.col(1) = b(0)*x % arma::cos(b(1) * x);
        A.col(2) = arma::sin(b(3) * x);
        A.col(3) = b(2)*x % arma::cos(b(3) * x);
        return A;
    };

    lsqr_opts opts;
    opts.jacobian_func = &J;
    lmlsqr(f, b_hat, opts);

    std::cout << "results after optimization : " << std::endl
              << "\tb_hat = " << b_hat.t()
              << "\tb_hat sum of squares = " << arma::norm(f(b_hat)) << std::endl << std::endl
              << "\tb = " << b.t()
              << "\tb sum of squares = " << arma::norm(f(b)) << std::endl << std::endl
              << "\t|error| = " << arma::norm(b_hat - b,"inf") << std::endl << std::endl
              << "\ttotal iterations required = " << opts.num_iters_returned << std::endl
              << "resulting model:" << std::endl
              << "\ty = " << b_hat(0) << " * sin(" << b_hat(1) << " * x) + " << b_hat(2) << " * cos(" << b_hat(3) << " * x)" << std::endl;
    std::cout << "We can, ofcourse, use lmlsqr to compute roots of an algebraic system, which is explored in newton_ex" << std::endl;

    arma::vec y_hat = b_hat(0)*arma::sin(b_hat(1)*t) + b_hat(2)*arma::sin(b_hat(3)*t);
    
    Gnuplot graph;
    stdv x1 = arma::conv_to<stdv>::from(x);
    stdv t1 = arma::conv_to<stdv>::from(t);
    stdv y_true1 = arma::conv_to<stdv>::from(y_true);
    stdv y1 = arma::conv_to<stdv>::from(y);
    stdv y_hat1 = arma::conv_to<stdv>::from(y_hat);

    graph.set_style("points");
    graph.plot_xy(x1,y1,"data");
    graph.set_style("lines");
    graph.plot_xy(t1,y_true1,"exact model");
    graph.plot_xy(t1,y_hat1,"least squares model");
    
    wait_for_key();

    return 0;
}