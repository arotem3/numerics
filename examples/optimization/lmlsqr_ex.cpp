#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o lmlsqr lmlsqr_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

typedef std::vector<double> dvec;

arma::vec f_true(const arma::vec& t) {
    return arma::exp(-t%t);
}

int main() {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "LmLSQR uses the Levenberg-Marquardt algorithm to find least squares solutions of root finding problems f(x) = 0" << std::endl;
    std::cout << "We will try to perform a nonlinear fit of the form:" << std::endl
              << "\ty = f(x) = b(0) + b(1) / (b(2) + b(3)*x^2)" << std::endl;
    
    arma::arma_rng::set_seed_random();
    double lower = -5; double upper = 5;
    arma::vec x = (upper-lower)*arma::randu(100) + lower;
    arma::vec t = arma::linspace(lower,upper);
    arma::vec y_true = f_true(t);
    arma::vec y = f_true(x) + 0.1*arma::randn(arma::size(x));

    arma::vec b = {-0.13217, 0.66178, 0.56353, 0.78157};
    arma::vec b_hat = {0,1,1,1}; b_hat += 0.1*arma::randn(4);
    std::cout << "we will use initial guess b_hat = ["
              << b_hat(0) << ", " << b_hat(1) << ", " << b_hat(2) << ", " << b_hat(3)
              << "] because it is close to our actual parameters" << std::endl << std::endl;

    auto f = [&x,&y](const arma::vec& b) -> arma::vec {
        return b(0) + b(1) / (b(2) + b(3)*x%x) - y;
    };

    // providing a jacobian function provides a substantial performance boost, you can try it with or without it.
    auto J = [&x](const arma::vec& b) -> arma::mat {
        arma::mat A(x.n_elem,4,arma::fill::zeros);
        A.col(0) = arma::ones(arma::size(x));
        A.col(1) = 1 / (b(2) + b(3)*x%x);
        A.col(2) = -b(1) / arma::pow(b(2) + b(3)*x%x, 2);
        A.col(3) = -b(1)*x%x / arma::pow(b(2) + b(3)*x%x, 2);
        return A;
    };

    numerics::optimization::LmLSQR lm(1e-3,100,true);
    lm.use_lu();
    // lm.fsolve(b_hat,f,J); // specify jacobian
    lm.fsolve(b_hat,f); // compute jacobian by finite differences and Broyden updates
    std::string flag = lm.get_exit_flag();

    std::cout << "results after optimization : " << std::endl
              << "\tb_hat = " << b_hat.t()
              << "\tb_hat sum of squares = " << arma::norm(f(b_hat)) << std::endl << std::endl
              << "\ttheoretical b = " << b.t()
              << "\ttheoretical b sum of squares  = " << arma::norm(f(b)) << std::endl
              << "\t||b - b_hat|| = " << arma::norm(b - b_hat,"inf") << std::endl << std::endl
              << "\tflag:" << flag << std::endl << std::endl
              << "resulting model:" << std::endl;
    std::cout << std::fixed << std::setprecision(3)
              << "\ty = " << b_hat(0) << " + " << b_hat(1) << " / (" << b_hat(2) << " + "  << b_hat(3) << "x^2)" << std::endl;
    std::cout << "We can, ofcourse, use lmlsqr to compute roots of an algebraic system, which is explored in newton_ex" << std::endl;

    arma::vec y_hat = b_hat(0) + b_hat(1) / (b_hat(2) + b_hat(3)*t%t);
    
    dvec xx = arma::conv_to<dvec>::from(x);
    dvec yy = arma::conv_to<dvec>::from(y);
    dvec tt = arma::conv_to<dvec>::from(t);
    dvec yyt = arma::conv_to<dvec>::from(y_true);
    dvec yyh = arma::conv_to<dvec>::from(y_hat);

    matplotlibcpp::named_plot("data",xx,yy,"og");
    matplotlibcpp::named_plot("exact model", tt, yyt, "--k");
    matplotlibcpp::named_plot("least squares model", tt, yyh, "-r");
    matplotlibcpp::legend();
    matplotlibcpp::show();

    return 0;
}