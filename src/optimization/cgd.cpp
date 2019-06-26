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

/* cgd(A,b,x,tol,max_iter) : solves the system Ax = b (or A'A*x = A'b) using conjugate gradient descent
 * --- A : system i.e. LHS (MAY BE OVERWITTEN IF 'A' IS NOT SYM POS DEF)
 * --- b : RHS (MAY BE OVERWRITTEN IF 'A' IS NOT SYM POS DEF)
 * --- x : initial guess and solution stored here
 * --- tol : stopping criteria for measuring convergence to solution.
 * --- max_iter : maximum number of iterations after which the solver will stop regardless of convergence */
void numerics::cgd(arma::mat& A, arma::mat& b, arma::mat& x, double tol, int max_iter) {
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

/* cgd(A,b,x,tol,max_iter) : solves the sparse system Ax = b (or A'A*x = A'b) using conjugate gradient descent
 * --- A : sparse system i.e. LHS (MAY BE OVERWITTEN)
 * --- b : RHS (MAY BE OVERWRITTEN)
 * --- x : initial guess and solution stored here
 * --- tol : stopping criteria for measuring convergence to solution.
 * --- max_iter : maximum number of iterations after which the solver will stop regardless of convergence */
void numerics::cgd(const arma::sp_mat& A, const arma::mat& b, arma::mat& x, double tol, int max_iter) {
    if (max_iter <= 0) max_iter = 1.1*b.n_rows;

    if (!A.is_symmetric()) {
        std::cerr << "cgd() error: sparse cgd() cannot handle nonsymmetric matrices." << std::endl;
        return;
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

/* minimize(grad_f, x, max_iter) : nonlinear conjugate gradient method.
 * --- grad_f : gradient function.
 * --- x : guess, and solution.
 * --- max_iter : maximum number of iterations after which the solver will stop regardless of convergence. */
void numerics::nlcgd::minimize(const std::function<arma::vec(const arma::vec&)>& grad_f, arma::vec& x, int max_iter) {
    if (max_iter <= 0) {
        if (max_iterations <= 0) max_iterations = 100;
    } else max_iterations = max_iter;

    bool minimize_line = (step_size <= 0);
    int n = x.n_elem;
    arma::vec p, s;
    double alpha = step_size, r, fval;

    uint k = 0;
    do {
        if (k >= max_iterations) {
            exit_flag = 1;
            num_iter += k;
            return;
        }
        p = -grad_f(x);
        if (p.has_nan() || p.has_inf()) {
            exit_flag = 2;
            num_iter += k;
            return;
        }
        s = p;
        r = 1/arma::norm(p,"inf");
        if (minimize_line) alpha = numerics::line_min(
            [&p,&x,&grad_f,r](double a) -> double {
                arma::vec z = x + (a*r)*p;
                return r*arma::dot( p, grad_f(z) );
            }
        );
        arma::vec ds = (alpha*r)*p - s;
        s = (alpha*r)*p;
        double ss = arma::dot(s,s);
        
        x+= s;
        k++;
        if (k%n == 0) {
            p = -grad_f(x);
        } else {
            double beta = std::max(arma::dot(s,ds)/ss, 0.0);
            p = beta*p - grad_f(x);
        }
    } while (1.0/r > tol);
    num_iter += k;
    exit_flag = 0;
}