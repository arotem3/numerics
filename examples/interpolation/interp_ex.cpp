#include "numerics.hpp"
#include "matplotlibcpp.h"

// g++ -g -Wall -o interp interp_ex.cpp -O3 -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

typedef std::vector<double> ddvec;

arma::vec f(const arma::vec& x) {
    return 0.5*x - 4*x%arma::exp(-arma::square(x))%(2*arma::square(x)-3);
}

int main() {
    arma::arma_rng::set_seed_random();
    double a = -3; double b = 3; double m = 20; double n = 300;

    arma::vec x = (b-a)*arma::regspace<arma::vec>(0,m)/m + a;
    arma::vec y = f(x);

    std::string extrapolation[5] = {
        "const",
        "boundary",
        "linear",
        "periodic",
        "polynomial"
    };

    arma::vec u = arma::linspace(-5, 5, n);
    arma::vec v;

    numerics::PieceWisePoly cspline = numerics::natural_cubic_spline(x, y, extrapolation[3]);
    numerics::PieceWisePoly hspline = numerics::hermite_cubic_spline(x,y, extrapolation[2]);

    matplotlibcpp::suptitle("interpolation");
    
    for (int i(0); i < 4; ++i) {
        std::string title;
        if (i==0) {v = cspline(u); title = "natural cubic spline with periodic extrapolation";}
        else if (i==1) {v = hspline(u); title = "Hermite spline with liear extrapolation";}
        else if (i==2) {v = numerics::lagrange_interp(x,y,u); title = "lagrange";}
        else if (i==3) {v = numerics::sinc_interp(x,y,u); title = "sinc";}

        ddvec xx = arma::conv_to<ddvec>::from(x);
        ddvec y1 = arma::conv_to<ddvec>::from(y.col(0));
        ddvec y2;
        if (y.n_cols==2) y2 = arma::conv_to<ddvec>::from(y.col(1));
        ddvec uu = arma::conv_to<ddvec>::from(u);
        ddvec v1 = arma::conv_to<ddvec>::from(v.col(0));
        ddvec v2;
        if (y.n_cols==2) v2 = arma::conv_to<ddvec>::from(v.col(1));

        matplotlibcpp::subplot(2,2,i+1);
        matplotlibcpp::title(title);
        matplotlibcpp::plot(xx, y1, "or");
        if (y.n_cols==2) matplotlibcpp::plot(xx, y2, "ob");
        
        matplotlibcpp::plot(uu, v1, "-r");
        if (y.n_cols==2) matplotlibcpp::plot(uu, v2, "-b");
        if (i > 0) matplotlibcpp::ylim(-5,5);
    }
    matplotlibcpp::tight_layout();
    matplotlibcpp::show();

    return 0;
}