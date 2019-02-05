#include "ODE.hpp"
#include "plot.hpp"

// g++ -Wall -g -o pois examples/poisson_ex.cpp examples/wait.cpp -lnumerics -larmadillo

using namespace ODE;
using namespace numerics;

arma::vec potential(const arma::vec& x, const arma::vec& y) {
    return 20*arma::sinc( 4*arma::pow(x-1, 2) + 4*arma::pow(y-2, 2)  );
}

void wait_for_key();

int main() {
    int num_pts = 24;

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

    Gnuplot fig;
    fig.set_title("2d poisson");

    plot3d(fig, soln.X, soln.Y, soln.U);

    wait_for_key();

    std::remove("data.csv");

    return 0;
}