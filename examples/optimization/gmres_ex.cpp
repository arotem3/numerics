#include "numerics.hpp"

// g++ -g -Wall -o gmres gmres_ex.cpp -O3 -lnumerics -larmadillo -lsuperlu


int main() {
    int n = 1000;
    arma::arma_rng::set_seed_random();
    arma::sp_mat A = arma::speye(n,n);
    A.diag(-1) = arma::randu(n-1)*0.5-0.25;
    A.diag(1)  = arma::randu(n-1)*0.5-0.25;
    arma::sp_mat B = arma::sprandu(n,n,0.05)*0.2;
    B.for_each([](arma::sp_mat::elem_type& elem)->void{elem -= 0.1;});
    A += B;
    // almost tridiagonal matrix.

    std::cout << "nonzeros / n = " << (double)A.n_nonzero / A.n_elem << std::endl;
    std::cout << "rcond(A) = " << arma::rcond(arma::mat(A)) << std::endl;

    arma::vec b = arma::randn(n); // we can solve multiple equations at once

    auto tic = std::chrono::high_resolution_clock::now();
    arma::vec x = arma::spsolve(A,b);
    auto toc = std::chrono::high_resolution_clock::now();
    double dur = std::chrono::duration_cast<std::chrono::milliseconds>(toc-tic).count()/1000.0;

    std::cout << "For the matrix of order " << n << ", the direct solver took " << dur << " seconds" << std::endl << std::endl;

    arma::vec y;
    tic = std::chrono::high_resolution_clock::now();
    numerics::optimization::gmres(y, A, b);
    toc = std::chrono::high_resolution_clock::now();
    dur = std::chrono::duration_cast<std::chrono::milliseconds>(toc-tic).count()/1000.0;

    std::cout << "gmres method took " << dur << " seconds" << std::endl
              << "the relative error was: " << arma::norm(A*y - b) / arma::norm(b) << std::endl << std::endl;

    return 0;
}