#include "../ODEs/ODE.hpp"
#include "gnuplot_i.hpp"

// g++ -Wall -g -o pois_ex examples/poisson_ex.cpp ODEs/cheb.cpp ODEs/poisson.cpp meshgrid.cpp examples/wait.cpp -larmadillo

using namespace ODE;
using namespace numerics;

arma::vec potential(const arma::vec& x, const arma::vec& y) {
    return 20*arma::sinc( 4*arma::pow(x-1, 2) + 4*arma::pow(y-2, 2)  );
}

void wait_for_key();

typedef std::vector<double> stdv;

int main() {
    int num_pts = 48;

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

    Gnuplot fig("2d poisson");
    fig.set_style("lines");

    for (int i(0); i < num_pts; ++i) {
        stdv x = arma::conv_to<stdv>::from(soln.X.row(i));
        stdv y = arma::conv_to<stdv>::from(soln.Y.row(i));
        stdv u = arma::conv_to<stdv>::from(soln.U.row(i));
        fig.plot_xyz(x,y,u);
    }

    wait_for_key();

    return 0;
}