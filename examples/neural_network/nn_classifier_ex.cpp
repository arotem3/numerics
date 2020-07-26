#include <numerics.hpp>

// g++ -g -Wall -O3 -o nn_classifier nn_classifier_ex.cpp -lnumerics -larmadillo

arma::uvec gen_classes(const arma::mat& x, int n_classes) {
    arma::mat xx = x + 0.1*arma::randn(arma::size(x));

    numerics::KMeansSGD km(n_classes*5);
    arma::uvec y = km.fit_predict(xx);
    arma::umat cc = arma::randperm(n_classes*5);
    cc.reshape(n_classes,5);
    std::map<arma::uword,arma::uword> cs;
    for (int i=0; i < 5; ++i) {
        for (int j=0; j < n_classes; ++j) {
            cs[cc(j,i)] = i;
        }
    }
    y.transform([&cs](arma::uword yi)->arma::uword{return cs[yi];});
    return y;
}

int main() {
    int N = 1000;
    arma::mat x = arma::randn(N, 2);
    
    arma::uvec y = gen_classes(x, 3);

    std::string loss = "categorical_crossentropy";
    std::vector<std::pair<int,std::string>> layers = {{100,"relu"}};
    long max_iter = 200;
    double tol = 1e-2;
    double l2 = 1e-4;
    double l1 = 0;
    std::string optimizer = "adam";
    bool verbose = true;

    numerics::NeuralNetClassifier model(layers, loss, max_iter, tol, l2, l1, optimizer, verbose);
    model.fit(x,y);

    std::cout << "accuracy : " << std::fixed << std::setprecision(2) << model.score(x,y) << "\n";
    return 0;
}
