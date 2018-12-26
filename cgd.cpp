#include "numerics.hpp"

//--- solves the system Ax = b using conjugate gradient descent ---//
void numerics::cgd(const arma::mat& A, const arma::vec& b, arma::vec& x, cg_opts& opts) {
    size_t n = opts.max_iter;
    if (n==0) n = 1.1*b.n_elem;
    
    arma::mat D;
    if (opts.is_symmetric && A.is_square()) D = A;
    else if (A.is_symmetric()) D = A;
    else D = A.t()*A;

    arma::vec r = b - D*x;
    arma::vec p;
    size_t k = 0;
    if (opts.preconditioner.empty()) { // not preconditioned
        p = r;
        while (arma::norm(r, "inf") > opts.err) {
            if (k >= n) { // too many iterations
                std::cerr << "cgd() warning: solution did not converge within the maximum allowed iterations." << std::endl
                          << "\treturning current best estimate" << std::endl
                          << "\tcurrent risidual: ||A*x - b|| = " << arma::norm(D*x - b) << std::endl;
                break;
            }
            double rtr = arma::dot(r,r);
            double alpha = rtr / arma::dot(p, D*p);
            x += alpha*p;
            r -= alpha*(D*p);
            double beta = arma::dot(r,r) / rtr;
            p = r + beta*p;
            k++;
        }
    } else { // preconditioned
        arma::mat Mi = opts.preconditioner.i();
        arma::vec z = Mi*r;
        p = z;
        while (arma::norm(r,"inf") > opts.err) {
            if (k >= n) { // too many iterations
                std::cerr << "cgd() warning: solution did not converge within the maximum allowed iterations." << std::endl
                          << "\treturning current best estimate" << std::endl
                          << "\tcurrent risidual: ||A*x - b|| = " << arma::norm(D*x - b) << std::endl;
                break;
            }
            double ztr = arma::dot(z,r);
            double alpha = ztr / arma::dot(p,D*p);
            x += alpha*p;
            r -= alpha*(D*p);
            z = Mi * r;
            double beta = arma::dot(z,r) / ztr;
            p = z + beta*p;
            k++;
        }
    }
    opts.num_iters_returned = k;
}

numerics::cg_opts numerics::cgd(const arma::mat& A, const arma::vec& b, arma::vec& x) {
    cg_opts opts;
    cgd(A,b,x,opts);
    return opts;
}

void numerics::sp_cgd(const arma::sp_mat& A, const arma::vec& b, arma::vec& x, cg_opts& opts) {
    size_t n = opts.max_iter;
    if (n==0) n = 1.1*b.n_elem;

    if (!opts.is_symmetric || !A.is_symmetric()) {
        std::cerr << "sp_cgd() error: sp_cgd() cannot handle nonsymmetric matrices" << std::endl;
        return;
    }

    arma::vec r = b - A*x;
    arma::vec p;
    size_t k = 0;
    if (opts.sp_precond == nullptr) { // not preconditioned
        p = r;
        while (arma::norm(r, "inf") > opts.err) {
            if (k >= n) { // too many iterations
                std::cerr << "cgd() warning: solution did not converge within the maximum allowed iterations." << std::endl
                          << "\treturning current best estimate" << std::endl
                          << "\tcurrent risidual: ||A*x - b|| = " << arma::norm(A*x - b) << std::endl;
                break;
            }
            double rtr = arma::dot(r,r);
            double alpha = rtr / arma::dot(p, A*p);
            x += alpha*p;
            r -= alpha*(A*p);
            double beta = arma::dot(r,r) / rtr;
            p = r + beta*p;
            k++;
        }
    } else { // preconditioned
        arma::vec z = opts.sp_precond->operator()(r);
        p = z;
        while (arma::norm(r,"inf") > opts.err) {
            if (k >= n) { // too many iterations
                std::cerr << "cgd() warning: solution did not converge within the maximum allowed iterations." << std::endl
                          << "\treturning current best estimate" << std::endl
                          << "\tcurrent risidual: ||A*x - b|| = " << arma::norm(A*x - b) << std::endl;
                break;
            }
            double ztr = arma::dot(z,r);
            double alpha = ztr / arma::dot(p,A*p);
            x += alpha*p;
            r -= alpha*(A*p);
            z = opts.sp_precond->operator()(r);
            double beta = arma::dot(z,r) / ztr;
            p = z + beta*p;
            k++;
        }
    }
    opts.num_iters_returned = k;
}

numerics::cg_opts numerics::sp_cgd(const arma::sp_mat& A, const arma::vec& b, arma::vec& x) {
    cg_opts opts;
    opts.is_symmetric = A.is_symmetric();
    sp_cgd(A,b,x,opts);
    return opts;
}