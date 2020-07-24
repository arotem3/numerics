#include "numerics.hpp"

// g++ -g -Wall -o cgd cgd_ex.cpp -O3 -lnumerics -larmadillo -lsuperlu


int main() {
    int n = 2000;
    arma::arma_rng::set_seed_random();
    arma::sp_mat A = arma::sprandn(5*n,n,0.001); // the sparser the system the greater the difference in performance b/w cgd() and spsolve()
    A = A.t() * A;
    std::cout << "nonzeros / n = " << (double)A.n_nonzero / A.n_elem << std::endl;

    arma::mat b = arma::randn(n,2); // we can solve multiple equations at once

    auto tic = std::chrono::high_resolution_clock::now();
    arma::mat x = arma::spsolve(A,b);
    auto toc = std::chrono::high_resolution_clock::now();
    double dur = std::chrono::duration_cast<std::chrono::milliseconds>(toc-tic).count()/1000.0;

    std::cout << "For the matrix of order " << n << ", the direct solver took " << dur << " seconds" << std::endl << std::endl;

    arma::mat y = arma::zeros(n,2);
    tic = std::chrono::high_resolution_clock::now();
    numerics::optimization::cgd(A, b, y);
    toc = std::chrono::high_resolution_clock::now();
    dur = std::chrono::duration_cast<std::chrono::milliseconds>(toc-tic).count()/1000.0;

    std::cout << "Conjugate gradient method took " << dur << " seconds" << std::endl
              << "the maximum cgd() error was: " << arma::norm(y-x,"inf") << std::endl << std::endl;

    return 0;
}