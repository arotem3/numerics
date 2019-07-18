#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o regularizer regularizer_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

using namespace numerics;
namespace plt = matplotlibcpp;

typedef std::vector<double> ddvec;

arma::mat f(arma::mat& X) {
    arma::mat y = arma::sign(X);
    return y;
}

arma::mat gen_basis(arma::vec& X, arma::vec& xx) {
    // radial basis
    arma::mat basis(xx.n_elem, X.n_elem);
    double h = arma::var(X)*2;
    for (uint i=0; i < X.n_elem; ++i) {
        basis.col(i) = arma::exp(-arma::square(xx-X(i))/h);
    }
    return basis;
}

arma::mat roughness_matrix(arma::vec& X, arma::vec& xx) {
    arma::mat C = arma::zeros(xx.n_elem, X.n_elem);
    for (uint i=0; i < X.n_elem; ++i) {
        // C.col(i) = 4*arma::square(xx - X(i)) - 2; // second derivative --> fewer wiggles
        C.col(i) = 2*(xx - X(i)); // first derivative --> flatter
    }
    C %= gen_basis(X,xx);
    return C.t() * C;
}

int main() {
    // arma::arma_rng::set_seed(123);
    int n_obs = 100;
    arma::vec X = 5*arma::randu(n_obs) - 2.5;
    arma::mat Y = f(X) + 0.05*arma::randn(n_obs,1);

    arma::mat basis = gen_basis(X,X);

    arma::mat c_overfit = arma::solve(basis, Y);
    arma::vec xgrid = arma::linspace(-3, 3, 200);
    arma::mat basis_grid = gen_basis(X,xgrid);
    arma::vec yhat_overfit = basis_grid * c_overfit;

    // regularizer lm(roughness_matrix(X,X)); std::cout << "...regularizing by regularization matrix, where we choose to use matrix measuring roughness." << std::endl << std::endl;
    // regularizer lm(0.01); std::cout << "...regularizing by explicit choice of lambda = 0.01 for L2 regularization." << std::endl << std::endl;
    regularizer lm; std::cout << "...regularizing by cross validation of lambdas for L2 regularization" << std::endl << std::endl;
    
    bool use_conjugate_gradiet = true;
    arma::vec c = lm.fit(basis, Y, use_conjugate_gradiet);
    arma::vec yhat = basis_grid*c;

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
    plt::ylim(-1.5, 1.5);
    plt::legend();
    plt::show();

    return 0;
}