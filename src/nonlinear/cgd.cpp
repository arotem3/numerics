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

/* CGD : solves the system Ax = b (or A'A*x = A'b) using conjugate gradient descent
 * --- A : system i.e. LHS (MAY BE OVERWITTEN IF A IS NOT SYM POS DEF)
 * --- b : RHS (MAY BE OVERWRITTEN IF A IS NOT SYM POS DEF)
 * --- x : initial guess and solution stored here
 * --- opts : options struct */
void numerics::cgd(arma::mat& A, arma::mat& b, arma::mat& x, cg_opts& opts) {
    uint n = opts.max_iter;
    if (n==0) n = 1.1*b.n_rows;
    
    if (!opts.is_symmetric || !A.is_symmetric()) {
        b = A.t()*b;
        A = A.t()*A;
    }

    arma::mat r = b - A*x;
    arma::mat p;
    uint k = 0;
    if (opts.preconditioner.is_empty() && opts.sp_precond.is_empty()) { // not preconditioned
        p = r;
        while (arma::norm(r, "inf") > r.n_cols * opts.err) {
            if (k >= n) { // too many iterations
                std::cerr << "cgd() warning: solution did not converge within the maximum allowed iterations." << std::endl
                          << "\treturning current best estimate" << std::endl
                          << "\tcurrent risidual: ||A*x - b|| = " << arma::norm(A*x - b,"fro") << std::endl;
                break;
            }
            arma::mat rtr = diag_inner_prod(r, r);
            arma::mat alpha = rtr * diag_inner_prod(p, A*p, true);
            x += p * alpha;
            r -= (A*p) * alpha;
            arma::mat beta = diag_inner_prod(r, r) * one_over_diag(rtr);
            p = r + p * beta;
            k++;
        }
    } else { // preconditioned
        arma::mat z;
        if (!opts.preconditioner.is_empty()) z = arma::solve(opts.preconditioner, r);
        else z = arma::spsolve(opts.sp_precond, r);
        p = z;
        while (arma::norm(r,"inf") > r.n_cols * opts.err) {
            if (k >= n) { // too many iterations
                std::cerr << "cgd() warning: solution did not converge within the maximum allowed iterations." << std::endl
                          << "\treturning current best estimate" << std::endl
                          << "\tcurrent risidual: ||A*x - b|| = " << arma::norm(A*x - b) << std::endl;
                break;
            }
            arma::mat ztr = diag_inner_prod(z, r);
            arma::mat alpha = ztr * diag_inner_prod(p, A*p, true);
            x += p * alpha;
            r -= (A*p) * alpha;

            if (!opts.preconditioner.is_empty()) z = arma::solve(opts.preconditioner, r);
            else z = arma::spsolve(opts.sp_precond, r);
            
            arma::mat beta = diag_inner_prod(z, r) * one_over_diag(ztr);
            p = z + p * beta;
            k++;
        }
    }
    opts.num_iters_returned = k;
}

/* CGD : solves the system Ax = b (or A'A*x = A'b) using conjugate gradient descent
 * --- A : system i.e. LHS (MAY BE OVERWITTEN)
 * --- b : RHS (MAY BE OVERWRITTEN)
 * --- x : initial guess and solution stored here */
numerics::cg_opts numerics::cgd(arma::mat& A, arma::mat& b, arma::mat& x) {
    cg_opts opts;
    cgd(A,b,x,opts);
    return opts;
}

/* SP_CGD : solves the sparse system Ax = b (or A'A*x = A'b) using conjugate gradient descent
 * --- A : sparse system i.e. LHS (MAY BE OVERWITTEN)
 * --- b : RHS (MAY BE OVERWRITTEN)
 * --- x : initial guess and solution stored here
 * --- opts : options struct */
void numerics::sp_cgd(const arma::sp_mat& A, const arma::mat& b, arma::mat& x, cg_opts& opts) {
    uint n = opts.max_iter;
    if (n==0) n = 1.1*b.n_cols;

    if (!opts.is_symmetric || !A.is_symmetric()) {
        std::cerr << "sp_cgd() error: sp_cgd() cannot handle nonsymmetric matrices." << std::endl;
        return;
    }

    arma::mat r = b - A*x;
    arma::mat p;
    uint k = 0;
    if (opts.sp_precond.is_empty() && opts.preconditioner.is_empty()) { // not preconditioned
        p = r;
        while (arma::norm(r, "inf") > r.n_cols * opts.err) {
            if (k >= n) { // too many iterations
                std::cerr << "cgd() warning: solution did not converge within the maximum allowed iterations." << std::endl
                          << "\treturning current best estimate" << std::endl
                          << "\tcurrent risidual: ||A*x - b|| = " << arma::norm(A*x - b,"fro") << std::endl;
                break;
            }
            arma::mat rtr = diag_inner_prod(r, r);
            arma::mat alpha = rtr * diag_inner_prod(p, A*p, true);
            x += p*alpha;
            r -= (A*p)*alpha;
            arma::mat beta = diag_inner_prod(r, r) * one_over_diag(rtr);
            p = r + p * beta;
            k++;
        }
    } else { // preconditioned
        arma::mat z;
        if (!opts.sp_precond.is_empty()) z = arma::spsolve(opts.sp_precond, r);
        else z = arma::solve(opts.preconditioner, r);
        p = z;
        while (arma::norm(r,"inf") > r.n_cols * opts.err) {
            if (k >= n) { // too many iterations
                std::cerr << "cgd() warning: solution did not converge within the maximum allowed iterations." << std::endl
                          << "\treturning current best estimate" << std::endl
                          << "\tcurrent risidual: ||A*x - b|| = " << arma::norm(A*x - b,"fro") << std::endl;
                break;
            }
            arma::mat ztr = diag_inner_prod(z, r);
            arma::mat alpha = ztr * diag_inner_prod(p, A * p, true);
            x += p * alpha;
            r -= (A*p) * alpha;

            if (!opts.sp_precond.is_empty()) z = arma::spsolve(opts.sp_precond, r);
            else z = arma::solve(opts.preconditioner, r);

            arma::mat beta = diag_inner_prod(z, r) * one_over_diag(ztr);
            p = z + p * beta;
            k++;
        }
    }
    opts.num_iters_returned = k;
}

/* SP_CGD : solves the sparse system Ax = b (or A'A*x = A'b) using conjugate gradient descent
 * --- A : sparse system i.e. LHS (MAY BE OVERWITTEN)
 * --- b : RHS (MAY BE OVERWRITTEN)
 * --- x : initial guess and solution stored here */
numerics::cg_opts numerics::sp_cgd(const arma::sp_mat& A, const arma::mat& b, arma::mat& x) {
    cg_opts opts;
    opts.is_symmetric = A.is_symmetric();
    sp_cgd(A,b,x,opts);
    return opts;
}