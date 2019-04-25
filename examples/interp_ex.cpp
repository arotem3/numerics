#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o interp examples/interp_ex.cpp -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

using namespace numerics;
namespace plt = matplotlibcpp;

typedef std::vector<double> ddvec;

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

    plt::suptitle("interpolation");
    
    for (int i(0); i < 5; ++i) {
        std::string title;
        if (i==0) {v = nearestInterp(x,y,u); title = "nearest neighbor";}
        else if (i==1) {v = linearInterp(x,y,u); title = "linear";}
        else if (i==2) {v = fSpline(u); title = "cubic spline";}
        else if (i==3) {v = lagrangeInterp(x,y,u); title = "lagrange";}
        else if (i==4) {v = sincInterp(x,y,u); title = "sinc";}

        std::cout << std::endl << title << std::endl;
        std::cout << "max error : " << arma::norm(v - f(u), "inf") << std::endl;

        ddvec xx = arma::conv_to<ddvec>::from(x);
        ddvec y1 = arma::conv_to<ddvec>::from(y.col(0));
        ddvec y2 = arma::conv_to<ddvec>::from(y.col(1));
        ddvec uu = arma::conv_to<ddvec>::from(u);
        ddvec v1 = arma::conv_to<ddvec>::from(v.col(0));
        ddvec v2 = arma::conv_to<ddvec>::from(v.col(1));

        plt::subplot(3,2,i+1);
        plt::title(title);
        plt::plot(xx, y1, "or");
        plt::plot(xx, y2, "ob");
        
        plt::plot(uu, v1, "-r");
        plt::plot(uu, v2, "-b");
        plt::ylim(-1.5,1.5);
    }
    plt::tight_layout();
    plt::show();

    return 0;
}