#include <numerics.hpp>
#include "matplotlibcpp.h"

// g++ -g -Wall -O3 -o nn_model model_ex.cpp -lnumerics -larmadillo -I/usr/include/python3.8 -lpython3.8

arma::mat f(arma::mat& X) {
    arma::mat y = arma::zeros(arma::size(X));
    for (int i=1; i < 10; ++i) {
        y += arma::sin(i*X)/i;
    }
    y = 0.5 - y/M_PI;
    return y;
}

using namespace numerics;
typedef std::vector<double> dvec;

int main() {
    int N = 1000;
    arma::mat x = 6*arma::randu(N,1) - 3;
    arma::vec y = f(x) + 0.2*arma::randn(arma::size(x));

    neuralnet::Layer input(1,15); // specify input shape and output shape
    input.name = "input layer"; // we can name layers for convinience
    input.set_activation("sqexp"); // specify activation by name

    neuralnet::Layer hidden(20); // specify only output shape, input shape is infered during compile()
    hidden.set_activation(neuralnet::Relu()); // specify activation by class which allows construction of custom activations that inherit from class Activation

    neuralnet::Layer output(1); // default activation is linear

    neuralnet::Model model(input);
    model.attach(hidden); // consecutively build model by attaching layers 
    model.attach(output);
    model.compile(); // infers model dimensions and initializes empty layer parameters

    model.set_loss(neuralnet::MAE()); // set by name (e.g. "mae") or class
    model.set_optimizer("adam"); // set by name or class
    model.set_l1(0.001); // set regularization for weights

    neuralnet::fit_parameters fitp; // set parameters for fit
    fitp.max_iter = 1000;
    fitp.tol = 1e-4;
    fitp.verbose = true;

    model.fit(x, y, fitp);

    neuralnet::Layer L = model.layers.at(1); // we can get a subview of the model, and perhaps copy over layers for use in another model

    arma::vec yh = model.predict(x);

    std::cout << "R^2 : " << std::fixed << std::setprecision(2) << r2_score(y, yh) << "\n";

    arma::vec t = arma::linspace(-3,3);
    yh = model.predict(t);

    dvec xx = arma::conv_to<dvec>::from(x);
    dvec yy = arma::conv_to<dvec>::from(y);
    dvec tt = arma::conv_to<dvec>::from(arma::vec(t));
    dvec yyh = arma::conv_to<dvec>::from(arma::vec(yh.col(0)));

    std::map<std::string,std::string> ls = {{"label","data"},{"ls","none"},{"marker","o"}};
    matplotlibcpp::plot(xx, yy, ls);
    ls["label"] = "fit"; ls["marker"] = ""; ls["ls"] = "-";
    matplotlibcpp::plot(tt, yyh, ls);
    matplotlibcpp::show();

    return 0;
}