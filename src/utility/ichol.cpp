#include "numerics.hpp"

/* ICHOL : incomplete Cholesky factorization of a square positive definite matrix.
 * --- A : matrix to perform Cholesky factorization on.
 * --- L : lower triangular matrix where, hopefully, A ~= L*L^T */
void numerics::ichol(const arma::mat& A, arma::mat& L) {
    if (!A.is_symmetric()) {
        std::cerr << "ichol() error: cannot perform incomplete Cholesky on an assymetric matrix." << std::endl;
        return;
    }
    int n = A.n_rows;
    L = A;
    for (int i=0; i < n; ++i) {
        L(i,i) = std::sqrt( L(i,i));
        for (int j=i+1; j < n; ++j) {
            if (L(j,i) != 0) L(j,i) = A(j,i)/L(i,i);
        }
        for (int j=i+1; j < n; ++j) {
            for (int k=j; k < n; ++k) {
                if (L(k,j) != 0) L(k,j) -= L(k,i)*L(j,i);
            }
        }
    }
    L = arma::trimatl(L);
}

/* ICHOL : incomplete Cholesky factorization of a square positive definite matrix.
 * --- A : matrix to perform Cholesky factorization on.
 * --- L : lower triangular matrix where, hopefully, A ~= L*L^T */
void numerics::ichol(const arma::sp_mat& A, arma::sp_mat& L) {
    if (!A.is_symmetric()) {
        std::cerr << "ichol() error: cannot perform incomplete Cholesky on an assymetric matrix." << std::endl;
        return;
    }
    int n = A.n_rows;
    L = A;
    for (int i=0; i < n; ++i) {
        L(i,i) = std::sqrt( L(i,i));
        for (int j=i+1; j < n; ++j) {
            if (L(j,i) != 0) L(j,i) = A(j,i)/L(i,i);
        }
        for (int j=i+1; j < n; ++j) {
            for (int k=j; k < n; ++k) {
                if (L(k,j) != 0) L(k,j) -= L(k,i)*L(j,i);
            }
        }
    }
    L = arma::trimatl(L);
}