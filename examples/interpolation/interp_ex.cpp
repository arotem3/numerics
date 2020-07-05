#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o interp interp_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python2.7 -lpython2.7

using namespace numerics;
namespace plt = matplotlibcpp;

typedef std::vector<double> ddvec;

arma::mat f(const arma::vec& x) {
    arma::mat y(x.n_elem, 2);
    y.col(0) = 0.5*arma::sin(2*x)%arma::exp(x/3);
    y.col(1) = arma::exp(-x%x);
    return y;
}

int main() {
    arma::arma_rng::set_seed_random();
    double a = -3; double b = 3; double m = 10; double n = 150; bool normalize_lagrange_interp = false;

    arma::vec x = (b-a)*arma::regspace<arma::vec>(0,m)/m + a;
    arma::mat y = f(x);

    cubic_interp cspline(x,y);
    arma::vec u = arma::linspace(x.min(), x.max(), n);
    arma::mat v;

    hspline_interp hspline(x,y);

    plt::suptitle("interpolation");
    
    for (int i(0); i < 4; ++i) {
        std::string title;
        if (i==0) {v = cspline(u); title = "cubic spline";}
        else if (i==1) {v = hspline(u); title = "Hermite spline";}
        else if (i==2) {v = lagrange_interp(x,y,u, normalize_lagrange_interp); title = "lagrange";}
        else if (i==3) {v = sinc_interp(x,y,u); title = "sinc";}

        std::cout << std::endl << title << std::endl;
        std::cout << "||error|| : " << arma::norm(v - f(u), "fro") << std::endl;

        ddvec xx = arma::conv_to<ddvec>::from(x);
        ddvec y1 = arma::conv_to<ddvec>::from(y.col(0));
        ddvec y2;
        if (y.n_cols==2) y2 = arma::conv_to<ddvec>::from(y.col(1));
        ddvec uu = arma::conv_to<ddvec>::from(u);
        ddvec v1 = arma::conv_to<ddvec>::from(v.col(0));
        ddvec v2;
        if (y.n_cols==2) v2 = arma::conv_to<ddvec>::from(v.col(1));

        plt::subplot(2,2,i+1);
        plt::title(title);
        plt::plot(xx, y1, "or");
        if (y.n_cols==2) plt::plot(xx, y2, "ob");
        
        plt::plot(uu, v1, "-r");
        if (y.n_cols==2) plt::plot(uu, v2, "-b");
        plt::ylim(-1.5,1.5);
    }
    plt::tight_layout();
    plt::show();

    return 0;
}