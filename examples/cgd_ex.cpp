#include "../numerics.hpp"

// g++ -g -Wall -o cgd_ex cgd.cpp examples/cgd_ex.cpp -larmadillo -lsuperlu

using namespace numerics;

int main() {
    int n = 1000;
    arma::arma_rng::set_seed_random();
    arma::sp_mat A = arma::sprandu(n,n,0.02); // the sparser the system the greater the difference in performance b/w cgd() and spsolve()
    A = (A + A.t())*0.5;
    A.diag() = n*arma::ones(n); // so we are guaranteed a symmetric positive definite matrix

    arma::vec b = arma::ones(n);

    clock_t t = clock();
    arma::vec x = arma::spsolve(A,b);
    t = clock() - t;

    std::cout << "For the matrix of order " << n << ", the direct solver took " << (float)t/CLOCKS_PER_SEC << " seconds" << std::endl;

    arma::vec y = arma::zeros(n);
    t = clock();
    sp_cgd(A,b,y);
    t = clock() - t;

    std::cout << "Conjugate gradient method took " << (float)t/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "the maximum error was: " << arma::norm(y-x,"inf") << std::endl;

    return 0;
}