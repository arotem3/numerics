#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o ridge ridge_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

using namespace numerics;
namespace plt = matplotlibcpp;

typedef std::vector<double> ddvec;

arma::mat f(arma::mat& X) {
    arma::mat y = arma::zeros(arma::size(X));
    for (int i=1; i < 10; ++i) {
        y += arma::sin(i*X)/i;
    }
    return y/M_PI;
}

arma::mat gen_basis(arma::vec& xx, uint np) {
    // radial basis
    arma::mat basis(xx.n_elem, np);
    for (uint i=0; i < np; ++i) {
        if (i==0) basis.col(i).fill(1);
        else basis.col(i) = arma::sin(M_PI*i*xx/5);
    }
    return basis;
}

int main() {
    // arma::arma_rng::set_seed(123);
    int n_obs = 100;
    arma::vec X = 5*arma::randu(n_obs) - 2.5;
    arma::mat Y = f(X) + 0.15*arma::randn(n_obs,1);

    uint n_poly = 50;
    arma::mat basis = gen_basis(X,n_poly);

    arma::mat c_overfit = arma::solve(basis, Y);
    arma::vec xgrid = arma::linspace(-2.5, 2.5, 500);
    arma::mat basis_grid = gen_basis(xgrid,n_poly);
    arma::vec yhat_overfit = basis_grid * c_overfit;

    numerics::ridge_cv lm;
    
    lm.fit(basis, Y);
    arma::vec c = lm.coef;
    arma::vec yhat = basis_grid*c;

    std::cout << "regularizing param: " << lm.regularizing_param << std::endl
              << "RMSE: " << lm.RMSE << std::endl
              << "effective degrees of freedom: " << lm.eff_df << std::endl;

    ddvec xx = arma::conv_to<ddvec>::from(X);
    ddvec yy = arma::conv_to<ddvec>::from(Y);
    ddvec xxg = arma::conv_to<ddvec>::from(xgrid);
    ddvec yoverfit = arma::conv_to<ddvec>::from(yhat_overfit);
    ddvec yyhat = arma::conv_to<ddvec>::from(yhat);

    plt::plot(xx,yy,"o");
    plt::named_plot("overfit", xxg, yoverfit);
    plt::named_plot("regularized fit", xxg, yyhat);
    plt::ylim(-1.2, 1.2);
    plt::legend();
    plt::show();

    return 0;
}