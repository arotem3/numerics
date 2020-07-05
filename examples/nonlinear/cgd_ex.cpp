#include "numerics.hpp"

// g++ -g -Wall -o cgd cgd_ex.cpp -O3 -lnumerics -larmadillo -lsuperlu

using namespace numerics;

int main() {
    int n = 1000;
    arma::arma_rng::set_seed_random();
    arma::sp_mat A = arma::sprandn(5*n,n,0.001); // the sparser the system the greater the difference in performance b/w cgd() and spsolve()
    A = A.t() * A;
    std::cout << "nonzeros / n = " << (double)A.n_nonzero / A.n_elem << std::endl;

    arma::mat b = arma::randn(n,2); // we can solve multiple equations at once

    clock_t t = clock();
    arma::mat x = arma::spsolve(A,b);
    t = clock() - t;

    std::cout << "For the matrix of order " << n << ", the direct solver took " << (float)t/CLOCKS_PER_SEC << " seconds" << std::endl << std::endl;

    arma::mat y = arma::zeros(n,2);
    t = clock();
    cgd(A, b, y);
    t = clock() - t;

    std::cout << "Conjugate gradient method took " << (float)t/CLOCKS_PER_SEC << " seconds" << std::endl
              << "the maximum cgd() error was: " << arma::norm(y-x,"inf") << std::endl << std::endl;

    arma::mat z;
    t = clock();
    linear_adj_gd(A,b,z);
    t = clock() - t;

    std::cout << "adjusted gradient descent took " << (float)t/CLOCKS_PER_SEC << " seconds" << std::endl
              << "the maximum linear_adj_gd() error was: " << arma::norm(z-x,"inf") << std::endl << std::endl;

    return 0;
}