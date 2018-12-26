#include "../numerics.hpp"
#include "gnuplot_i.hpp"

void wait_for_key(std::string s);

using namespace numerics;

typedef std::vector<double> stdv;

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

    Gnuplot fig("interpolation examples");
    fig.set_yrange(-1.5,1.5);
    stdv x0 = arma::conv_to<stdv>::from(x);
    stdv y0 = arma::conv_to<stdv>::from(y.col(0));
    stdv y1 = arma::conv_to<stdv>::from(y.col(1));
    stdv u0 = arma::conv_to<stdv>::from(u);
    
    for (int i(0); i < 5; ++i) {
        std::string title;
        if (i==0) {v = nearestInterp(x,y,u); title = "nearest neighbor interpolation.";}
        else if (i==1) {v = linearInterp(x,y,u); title = "linear interpolation";}
        else if (i==2) {v = fSpline(u); title = "cubic spline interpolation";}
        else if (i==3) {v = lagrangeInterp(x,y,u); title = "lagrange interpolation";}
        else if (i==4) {v = sincInterp(x,y,u); title = "sinc interpolation";}

        std::cout << std::endl << title << std::endl;
        std::cout << "max error : " << arma::norm(v - f(u), "inf") << std::endl;

        stdv v0 = arma::conv_to<stdv>::from(v.col(0));
        stdv v1 = arma::conv_to<stdv>::from(v.col(1));

        fig.set_style("points");
        fig.plot_xy(x0,y0,"original x,y");
        fig.plot_xy(x0,y1,"original x,y");
        fig.set_style("lines");
        fig.plot_xy(u0,v0,title);
        fig.plot_xy(u0,v1,title);

        wait_for_key("Press ENTER for next example...");
        fig.reset_plot();
    }
    return 0;
}