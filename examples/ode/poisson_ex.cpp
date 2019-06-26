#include <numerics.hpp>
#include "matplotlibcpp.h"

// g++ -Wall -g -o pois poisson_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

typedef std::vector<std::vector<double>> ddvec;

arma::vec potential(const arma::vec& x, const arma::vec& y) {
    return 20*arma::sinc( 4*arma::pow(x-1, 2) + 4*arma::pow(y-2, 2)  );
}

using namespace numerics::ode;

int main(int num_pts = 32) {
    // int num_pts = 32;

    boundary_conditions_2d bc;
    bc.lower_x = -1;
    bc.lower_y = 0;
    bc.upper_x = 3;
    bc.upper_y = 4;
    bc.dirichlet_condition = [](const arma::mat& x, const arma::mat& y) -> arma::mat {
        arma::mat B = arma::zeros(arma::size(x));
        B( arma::find(x==-1) ).zeros();
        B( arma::find(y==0) ) = arma::sin(M_PI*x(arma::find(y==0)));
        B( arma::find(x==3) ) = arma::sinc(2*y(arma::find(x==3)) - 4);
        B( arma::find(y==4) ) = -0.5*arma::exp( -arma::pow(x(arma::find(y==4))-0.75,2)*5 );
        return B;
    };
    arma::mat X, Y, U;
    poisson2d(X, Y, U, potential, bc, num_pts);
    std::cout << X << Y << U << std::endl;

    ddvec xx(num_pts), yy(num_pts), zz(num_pts);
    for (uint i=0; i < X.n_rows; ++i) {
        for (uint j=0; j < Y.n_cols; ++j) {
            xx.at(i).push_back(X(i,j));
            xx.at(i).push_back(Y(i,j));
            xx.at(i).push_back(U(i,j));
        }
    }

    matplotlibcpp::plot_surface(xx,yy,zz);
    matplotlibcpp::title("2D Poisson");
    matplotlibcpp::show();

    return 0;
}