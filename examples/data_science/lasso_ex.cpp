#include "numerics.hpp"
#include "matplotlibcpp.h"


// g++ -g -Wall -o lasso lasso_ex.cpp -larmadillo -lnumerics -I/usr/include/python2.7 -lpython2.7

int main() {
    int n=100;
    arma::arma_rng::set_seed(123);
    arma::vec x = 2*arma::randu(n)-1;
    arma::vec y = 1 - 0.1*x + x%x + 0.3*arma::randn(n); // random data is quadratic in x...

    arma::mat P(n,31); // but we don't know that so let's fit a 30 degree polynomial (including an intercept)
    for (int i=0; i < 31; ++i) P.col(i) = arma::pow(x,i);
    
    numerics::lasso_cv lcv;
    lcv.fit(P,y,true);
    arma::vec w = lcv.coef;
    std::cout << "optimal lambda: " << lcv.lambda << "\n";

    // print + plot results
    std::stringstream model;
    model << "model : y = " << std::fixed << std::setprecision(3) << w(0);
    for (int i=1; i < 31; ++i) if (std::abs(w(i)) > 1e-4) model << " + " << w(i) << "*x^" << i;
    std::string model_str = model.str();
    std::cout << model_str + "\n";

    arma::vec t = arma::linspace(x.min(), x.max());
    arma::mat Pt(100,31);
    for (int i=0; i < 31; ++i) Pt.col(i) = arma::pow(t,i);

    std::vector<double> xx = arma::conv_to<std::vector<double>>::from(x);
    std::vector<double> yy = arma::conv_to<std::vector<double>>::from(y);
    std::vector<double> tt = arma::conv_to<std::vector<double>>::from(t);
    std::vector<double> pp = arma::conv_to<std::vector<double>>::from(Pt*w);
    std::vector<double> ptrue = arma::conv_to<std::vector<double>>::from(1 - 0.1*t+t%t);

    matplotlibcpp::named_plot("data", xx, yy, ".b");
    matplotlibcpp::named_plot("true function", tt, ptrue, "--k");
    matplotlibcpp::named_plot("predicted function", tt, pp, "-r");
    matplotlibcpp::legend();
    matplotlibcpp::title(model_str);
    matplotlibcpp::show();

    return 0;
}