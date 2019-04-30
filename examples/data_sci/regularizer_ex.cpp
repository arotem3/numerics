#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o regularizer regularizer_ex.cpp -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

using namespace numerics;
namespace plt = matplotlibcpp;

typedef std::vector<double> ddvec;

arma::mat f(arma::mat& X) {
    arma::mat y = arma::zeros(arma::size(X));
    for (int i=1; i < 10; ++i) {
        y += arma::sin(i*X)/i;
    }
    return 0.5 - y/M_PI;
}

arma::mat gen_basis(arma::mat& X, arma::vec& centers) {
    arma::mat basis = arma::zeros(X.n_elem, centers.n_elem);
    for (uint i=0; i < centers.n_elem; ++i) {
        basis.col(i) = arma::exp(-arma::square(X-centers(i)));
    }
    return basis;
}

arma::mat roughness_matrix(arma::mat& X, arma::vec& centers) {
    arma::mat C = arma::zeros(X.n_elem, centers.n_elem);
    for (uint i=0; i < centers.n_elem; ++i) {
        C.col(i) = X - centers(i);
    }
    C %= gen_basis(X, centers);
    return 0.01 * C.t() * C;
}

int main() {
    // arma::arma_rng::set_seed(123);
    int n_obs = 40;
    arma::vec X = 5*arma::randu(n_obs) - 2.5;
    arma::mat Y = f(X) + 0.05*arma::randn(n_obs,1);

    int N = 30;
    arma::vec centers = arma::linspace(-2.5, 2.5, N);
    arma::mat basis = gen_basis(X, centers);

    arma::mat c_overfit = arma::solve(basis, Y);
    arma::vec xgrid = arma::linspace(-3, 3, 200);
    arma::mat basis_grid = gen_basis(xgrid, centers);
    arma::vec yhat_overfit = basis_grid * c_overfit;

    std::cout << "the regularizer class has three options for model selection" << std::endl
              << "\t(1) Provide a regularization matrix which is used if you have additional information about your model such as smoothness" << std::endl
              << "\t(2) Provide a lambda parameter for L2 regularization where the cost is given by (y - X*c)^2 + lambda * c^2" << std::endl
              << "\t(3) Let the model select a lambda for L2 regularization by cross validation." << std::endl << std::endl; 

    // regularizer lm(roughness_matrix(X, centers)); std::cout << "...regularizing by regularization matrix, where we choose to use matrix measuring roughness." << std::endl << std::endl;
    // regularizer lm(0.01); std::cout << "...regularizing by explicit choice of lambda = 0.01 for L2 regularization." << std::endl << std::endl;
    regularizer lm; std::cout << "...regularizing by cross validation of lambdas for L2 regularization" << std::endl << std::endl;
    
    arma::vec yhat = lm.fit(basis, Y).predict(basis_grid);

    std::cout << "regularizing param: " << lm.regularizing_param() << std::endl
              << "MSE: " << lm.MSE() << std::endl
              << "effective degrees of freedom: " << lm.eff_df() << std::endl;

    ddvec xx = arma::conv_to<ddvec>::from(X);
    ddvec yy = arma::conv_to<ddvec>::from(Y);
    ddvec xxg = arma::conv_to<ddvec>::from(xgrid);
    ddvec yoverfit = arma::conv_to<ddvec>::from(yhat_overfit);
    ddvec yyhat = arma::conv_to<ddvec>::from(yhat);

    plt::plot(xx,yy,"o");
    plt::named_plot("overfit", xxg, yoverfit);
    plt::named_plot("regularized fit", xxg, yyhat);
    plt::ylim(-0.2, 1.2);
    plt::legend();
    plt::show();

    return 0;
}