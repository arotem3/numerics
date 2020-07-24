#include "numerics.hpp"

arma::mat diag_inner_prod(const arma::mat& A, const arma::mat& B, bool one_over = false) {
    arma::mat P = arma::zeros(A.n_cols, A.n_cols);
    for (uint i=0; i < A.n_cols; ++i) {
        P(i,i) = arma::dot(A.col(i), B.col(i));
        if (one_over && P(i,i)!=0) P(i,i) = 1/P(i,i);
        else if (one_over && P(i,i)==0) P(i,i) = 0;
    }
    return P;
}

arma::mat& one_over_diag(arma::mat& A) {
    for (uint i=0; i < A.n_rows; ++i) {
        if (A(i,i) != 0) A(i,i) = 1/A(i,i);
    }
    return A;
}

void numerics::optimization::cgd(arma::mat& A, arma::mat& b, arma::mat& x, double tol, int max_iter) {
    if (max_iter <= 0) max_iter = 1.1*b.n_rows;
    
    if ( !A.is_symmetric() ) {
        b = A.t()*b;
        A = A.t()*A;
    }
    if (x.empty()) x = arma::randn(A.n_cols,b.n_cols);

    arma::mat r = b - A*x;
    arma::mat p;
    uint k = 0;
    p = r;
    while (arma::norm(r, "inf") > tol) {
        if (k >= max_iter) break;
        arma::mat rtr = diag_inner_prod(r, r);
        arma::mat alpha = rtr * diag_inner_prod(p, A*p, true);
        x += p * alpha;
        r -= (A*p) * alpha;
        arma::mat beta = diag_inner_prod(r, r) * one_over_diag(rtr);
        p = r + p * beta;
        k++;
    }
}

void numerics::optimization::cgd(const arma::sp_mat& A, const arma::mat& b, arma::mat& x, double tol, int max_iter) {
    if (max_iter <= 0) max_iter = 1.1*b.n_rows;

    if (!A.is_symmetric()) {
        throw std::invalid_argument("sparse cgd cannot handle nonsymmetric matrices.");
    }

    arma::mat r = b - A*x;
    arma::mat p;
    uint k = 0;
    p = r;
    while (arma::norm(r, "inf") > tol) {
        if (k >= max_iter) break;
        arma::mat rtr = diag_inner_prod(r, r);
        arma::mat alpha = rtr * diag_inner_prod(p, A*p, true);
        x += p*alpha;
        r -= (A*p)*alpha;
        arma::mat beta = diag_inner_prod(r, r) * one_over_diag(rtr);
        p = r + p * beta;
        k++;
    }
}