#include "numerics.hpp"

// g++ -g -Wall -o pcg pcg_ex.cpp -O3 -lnumerics -larmadillo -lsuperlu


int main() {
    int n = 2000;
    arma::arma_rng::set_seed_random();
    arma::sp_mat A = arma::sprandn(5*n,n,0.001); // the sparser the system the greater the difference in performance b/w pcg() and spsolve()
    A = A.t() * A;
    std::cout << "nonzeros / n = " << (double)A.n_nonzero / A.n_elem << std::endl;

    arma::vec b = arma::randn(n); // we can solve multiple equations at once

    auto tic = std::chrono::high_resolution_clock::now();
    arma::mat x = arma::spsolve(A,b);
    auto toc = std::chrono::high_resolution_clock::now();
    double dur = std::chrono::duration_cast<std::chrono::milliseconds>(toc-tic).count()/1000.0;

    std::cout << "For the matrix of order " << n << ", the direct solver took " << dur << " seconds" << std::endl << std::endl;

    arma::vec y;
    tic = std::chrono::high_resolution_clock::now();
    numerics::optimization::pcg(y, A, b);
    toc = std::chrono::high_resolution_clock::now();
    dur = std::chrono::duration_cast<std::chrono::milliseconds>(toc-tic).count()/1000.0;

    std::cout << "Conjugate gradient method took " << dur << " seconds" << std::endl
              << "the relative error was: " << arma::norm(A*y - b) / arma::norm(b) << std::endl << std::endl;

    return 0;
}