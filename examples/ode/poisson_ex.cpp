#include "ODE.hpp"
#include "matplotlibcpp.h"

// g++ -Wall -g -o pois poisson_ex.cpp -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

using namespace ODE;
using namespace numerics;
typedef std::vector<std::vector<double>> ddvec;

arma::vec potential(const arma::vec& x, const arma::vec& y) {
    return 20*arma::sinc( 4*arma::pow(x-1, 2) + 4*arma::pow(y-2, 2)  );
}

int main() {
    int num_pts = 36;

    bcfun_2d bc;
    bc.lower_x = -1;
    bc.lower_x_bc = [](const arma::vec& y) -> arma::vec {return arma::zeros(arma::size(y));}; // u(-1,y) = 0

    bc.lower_y = 0;
    bc.lower_y_bc = [](const arma::vec& x) -> arma::vec {return 0.1*arma::sin(M_PI*x);}; // u(x,0) = sin(pi*x)

    bc.upper_x = 3;
    bc.upper_x_bc = [](const arma::vec& y) -> arma::vec {return arma::sinc(2*(y-2));}; // u(3,y) = sinc(2*(y-2))

    bc.upper_y = 4;
    bc.upper_y_bc = [](const arma::vec& x) -> arma::vec {return -0.5*arma::exp( -arma::pow(x-0.75,2)*5 );}; // u(x,4) = -exp(-5*(x-0.75)^2)/2

    soln_2d soln = poisson2d(potential, bc, num_pts);

    /* std::ofstream data("pois_soln.txt");
    soln.save(data);
    data.close(); */

    ddvec X(num_pts), Y(num_pts), Z(num_pts);
    for (int i=0; i < num_pts; ++i) {
        for (int j=0; j < num_pts; ++j) {
            X.at(i).push_back(soln.X(i,j));
            Y.at(i).push_back(soln.Y(i,j));
            Z.at(i).push_back(soln.U(i,j));
        }
    }

    matplotlibcpp::plot_surface(X,Y,Z);
    matplotlibcpp::title("2D Poisson");
    matplotlibcpp::show();

    return 0;
}