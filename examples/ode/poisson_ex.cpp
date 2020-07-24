#include <numerics.hpp>
#include "matplotlibcpp.h"

// g++ -Wall -g -o pois poisson_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

typedef std::vector<std::vector<double>> ddvec;

arma::vec potential(const arma::vec& x, const arma::vec& y) {
    return 20*arma::sinc( 4*arma::pow(x-1, 2) + 4*arma::pow(y-2, 2)  );
}

arma::mat bc(const arma::mat& x, const arma::mat& y) {
    arma::mat B = arma::zeros(arma::size(x));
    B( arma::find(x==-1) ).zeros();
    B( arma::find(y==0) ) = arma::sin(M_PI*x(arma::find(y==0)));
    B( arma::find(x==3) ) = arma::sinc(2*y(arma::find(x==3)) - 4);
    B( arma::find(y==4) ) = -0.5*arma::exp( -arma::pow(x(arma::find(y==4))-0.75,2)*5 );
    return B;
    // return 0*x;
}

using namespace numerics::ode;

int main() {
    int num_pts = 32;
    double root_eig = 5.5;

    arma::mat X = {-1,3},
              Y = {0,2};

    arma::mat U;

    poisson_helmholtz_2d(X, Y, U, potential, bc, root_eig, num_pts);

    ddvec xx(num_pts), yy(num_pts), zz(num_pts);
    for (uint i=0; i < X.n_rows; ++i) {
        for (uint j=0; j < Y.n_cols; ++j) {
            xx.at(i).push_back(X(i,j));
            yy.at(i).push_back(Y(i,j));
            zz.at(i).push_back(U(i,j));
        }
    }

    std::map<std::string,std::string> keys;
    keys["cmap"] = "plasma";
    matplotlibcpp::plot_surface(xx,yy,zz,keys);
    matplotlibcpp::title("2D Poisson");
    matplotlibcpp::show();

    return 0;
}