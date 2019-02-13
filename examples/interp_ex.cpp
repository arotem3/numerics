#include "numerics.hpp"
#include "plot.hpp"

// g++ -g -Wall -o interp examples/interp_ex.cpp examples/wait.cpp -lnumerics -larmadillo

void wait_for_key(std::string s);

using namespace numerics;

arma::mat f(const arma::vec& x) {
    arma::mat y(x.n_elem, 2);
    // y = arma::sign(x); // step function
    // y = arma::round(arma::sin(x*M_PI)); // square wave
    y.col(0) = 0.5*arma::sin(x%x)%arma::exp(x/3);
    y.col(1) = arma::exp(-x%x);
    // y = arma::zeros(arma::size(x)); arma::uvec a = arma::find(x == 0); y(a) = arma::ones(arma::size(a));
    // y = arma::zeros(arma::size(x)); arma::uvec a = arma::find(arma::abs(x) <= 1); y(a) = 1 - arma::abs(x(a));
    return y;
}

int main() {
    double a = -3; double b = 3; double m = 20; double n = 150;

    arma::vec x = (b-a)*arma::regspace<arma::vec>(0,m)/m + a;
    arma::mat y = f(x);

    CubicInterp fSpline(x,y);
    arma::vec u = (b-a)*arma::regspace<arma::vec>(0,n)/n + a;
    arma::mat v;

    Gnuplot fig;
    fig.set_yrange(-1.5,1.5);
    
    for (int i(0); i < 5; ++i) {
        std::string title;
        if (i==0) {v = nearestInterp(x,y,u); title = "nearest neighbor interpolation.";}
        else if (i==1) {v = linearInterp(x,y,u); title = "linear interpolation";}
        else if (i==2) {v = fSpline(u); title = "cubic spline interpolation";}
        else if (i==3) {v = lagrangeInterp(x,y,u); title = "lagrange interpolation";}
        else if (i==4) {v = sincInterp(x,y,u); title = "sinc interpolation";}

        std::cout << std::endl << title << std::endl;
        std::cout << "max error : " << arma::norm(v - f(u), "inf") << std::endl;

        plot(fig, x, (arma::mat)y.col(0), {{"legend","original x,y"},{"linespec","or"}});
        plot(fig, x, (arma::mat)y.col(1), {{"legend","original x,y"},{"linespec","ob"}});

        plot(fig, u, (arma::mat)v.col(0), {{"legend",title},{"linespec","-r"}});
        plot(fig, u, (arma::mat)v.col(1), {{"legend",title},{"linespec","-b"}});

        wait_for_key("Press ENTER for next example...");
        fig.reset_plot();
    }
    return 0;
}