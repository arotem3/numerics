#include "numerics.hpp"
#include "matplotlibcpp.h"


// g++ -g -Wall -o lasso lasso_ex.cpp -larmadillo -lnumerics -I/usr/include/python3.8 -lpython3.8

int main() {
    int n=200;
    arma::arma_rng::set_seed(123);
    arma::vec x = 2*arma::randu(n)-1;
    arma::vec y = 1 - 0.1*x + x%x + 0.3*arma::randn(n); // random data is quadratic in x...

    numerics::PolyFeatures poly(20);
    arma::mat P = poly.fit_predict(x);
    
    numerics::LassoCV lcv;
    lcv.fit(P,y);
    arma::vec w = lcv.linear_coefs;
    std::cout << "optimal lambda: " << lcv.lambda << "\n";

    // print + plot results
    std::stringstream model;
    model << "model : y = " << std::fixed << std::setprecision(3) << lcv.intercept;
    for (uint i=0; i < w.n_elem; ++i) if (w(i) != 0) model << " + " << w(i) << "*x^" << i+1;
    std::string model_str = model.str();
    std::cout << model_str + "\n";

    arma::vec t = arma::linspace(x.min(), x.max());
    arma::mat Pt = poly.predict(t);
    arma::vec yh = lcv.predict(Pt);

    std::vector<double> xx = arma::conv_to<std::vector<double>>::from(x);
    std::vector<double> yy = arma::conv_to<std::vector<double>>::from(y);
    std::vector<double> tt = arma::conv_to<std::vector<double>>::from(t);
    std::vector<double> pp = arma::conv_to<std::vector<double>>::from(yh);
    std::vector<double> ptrue = arma::conv_to<std::vector<double>>::from(1 - 0.1*t+t%t);

    matplotlibcpp::named_plot("data", xx, yy, ".b");
    matplotlibcpp::named_plot("true function", tt, ptrue, "--k");
    matplotlibcpp::named_plot("predicted function", tt, pp, "-r");
    matplotlibcpp::legend();
    matplotlibcpp::title(model_str);
    matplotlibcpp::show();

    return 0;
}