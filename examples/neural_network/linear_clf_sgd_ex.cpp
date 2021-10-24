#include <numerics.hpp>

// g++ -g -Wall -O3 -o sgd_classifier linear_clf_sgd_ex.cpp -lnumerics -larmadillo

arma::uvec gen_classes(const arma::mat& x, int n_classes) {
    arma::mat xx = x + 2.5*arma::randn(arma::size(x));

    numerics::KMeansSGD km(n_classes);
    return km.fit_predict(xx);
}

int main() {
    int N = 10000;
    arma::mat x = arma::randn(N, 5);
    

    arma::uvec y = gen_classes(x, 3);

    std::string loss = "categorical_crossentropy";
    long max_iter = 200;
    double tol = 1e-4;
    double l2 = 1e-4;
    double l1 = 0;
    std::string optimizer = "adam";
    bool verbose = true;

    numerics::LinearClassifierSGD model(loss, max_iter, tol, l2, l1, optimizer, verbose);
    model.fit(x,y);

    std::cout << "accuracy : " << std::fixed << std::setprecision(2) << model.score(x,y) << "\n";
    return 0;
}