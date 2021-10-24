#include <numerics.hpp>

// g++ -g -Wall -O3 -o sgd_regression linear_reg_sgd_ex.cpp -lnumerics -larmadillo

int main() {
    int N = 10000;
    arma::mat x = arma::randn(N, 5);
    arma::vec w = 10*arma::randu(5)-5;
    double b = 3.14;

    arma::vec y = x*w + b + 2.5*arma::randn(N);

    std::string loss = "mse";
    long max_iter = 200;
    double tol = 1e-4;
    double l2 = 1e-4;
    double l1 = 0;
    std::string optimizer = "adam";
    bool verbose = true;

    numerics::LinearRegressorSGD model(loss, max_iter, tol, l2, l1, optimizer, verbose);
    model.fit(x,y);

    std::cout << "R2 : " << std::fixed << std::setprecision(2) << model.score(x,y) << "\n";
    return 0;
}