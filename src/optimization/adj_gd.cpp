#include "numerics.hpp"

/* linear_adj_gd(A, b, x, tol, max_iter) : like adj_gd by for linear systems A*x = b
 * --- A, b, x : A*x = b. A and b may be overwritten if A is not symmetric. x will be set to the solution.
 * --- tol : stopping criteria for itteration. */
void numerics::linear_adj_gd(arma::mat& A, arma::mat& b, arma::mat& x, double tol, int max_iter) {
    if (!A.is_symmetric()) {
        b = A.t()*b;
        A = A.t()*A;
    }
    if (max_iter <= 0) max_iter = b.n_rows;
    if (x.empty()) x = arma::randn(arma::size(b));

    int k = 1;
    arma::mat g, v, prev_x;
    do {
        if (k >= max_iter) break;
        if (k%3==1 || k%3==2) {
            g = A*x - b;
            v = g;
            if (k%3==1) prev_x = x;
        } else { // every third itteration
            g = prev_x - x;
            v = A*x - b;
        }
        arma::rowvec alpha(b.n_cols);
        for (int i=0; i < b.n_cols; ++i) {
            alpha(i) = (arma::dot(b.col(i),v.col(i)) - arma::dot(x.col(i),A*v.col(i))) / arma::dot(g.col(i),A*v.col(i));
        }
        if (k%3==0) {
            // since we are using a multistep method, we need to check stability. stability polynomial = (r-1)*(r^2 + alpha(r+1))
            for (int i=0; i < b.n_cols; ++i) {
                if (alpha(i) > -0.5) {
                    x.col(i) += alpha(i) * g.col(i);
                    k++;
                }
            }
        } else {
            for (int i=0; i < b.n_cols; ++i) {
                x.col(i) += alpha(i) * g.col(i);
                k++;
            }
        }
    } while (arma::norm(v,"fro") > tol);
}

/* linear_adj_gd(A, b, x, tol, max_iter) : like adj_gd by for linear systems A*x = b
 * --- A, b, x : A*x = b. x will be set to the solution.
 * --- tol : stopping criteria for itteration. */
void numerics::linear_adj_gd(const arma::sp_mat& A, const arma::mat& b, arma::mat& x, double tol, int max_iter) {
    if (!A.is_symmetric()) {
        std::cerr << "linear_adj_gd() error: sparse linear_adj_gd() cannot handle nonsymmetric matrices." << std::endl;
    }
    if (max_iter <= 0) max_iter = b.n_rows;
    if (x.empty()) x = arma::randn(arma::size(b));

    int k = 1;
    arma::mat g, v, prev_x;
    do {
        if (k >= max_iter) break;

        if (k%3==1 || k%3==2) {
            g = A*x - b;
            v = g;
            if (k%3==1) prev_x = x;
        } else { // every third itteration
            g = prev_x - x;
            v = A*x - b;
        }
        arma::rowvec alpha(b.n_cols);
        for (int i=0; i < b.n_cols; ++i) {
            alpha(i) = (arma::dot(b.col(i),v.col(i)) - arma::dot(x.col(i),A*v.col(i))) / arma::dot(g.col(i),A*v.col(i));
        }
        if (k%3==0) {
            // since we are using a multistep method, we need to check stability. stability polynomial = (r-1)*(r^2 + alpha(r+1))
            for (int i=0; i < b.n_cols; ++i) {
                if (alpha(i) > -0.5) {
                    x.col(i) += alpha(i) * g.col(i);
                    k++;
                }
            }
        } else {
            for (int i=0; i < b.n_cols; ++i) {
                x.col(i) += alpha(i) * g.col(i);
                k++;
            }
        }
    } while (arma::norm(v,"fro") > tol);
}


/*  minimize(grad_f, x) : "adjusted" gradient descent, a multistep acceleration method of my own design
 * --- grad_f : gradient function.
 * --- x : intial guess and output variable. */
void numerics::adj_gd::minimize(std::function<arma::vec(const arma::vec&)> grad_f, arma::vec& x, int max_iter) {
    if (max_iter <= 0) {
        if (max_iterations <= 0) max_iterations = 100;
    } else max_iterations = max_iter;

    bool minimize_line = (step_size <= 0);

    double alpha = step_size, r, fval;
    arma::vec p, prev_x;

    uint k = 1;
    do {
        if (k > max_iterations) {
            exit_flag = 1;
            num_iter += k;
            return;
        }

        if (k%3==1 || k%3==2) {
            p = -grad_f(x);
            if (k%3==1) prev_x = x;
        } else p = prev_x - x;

        if (p.has_nan() || p.has_inf()) {
            exit_flag = 2;
            num_iter += k;
            return;
        }

        r = arma::norm(p,"inf");
        if (minimize_line) alpha = numerics::line_min(
            [&p,&x,&grad_f,r](double a) -> double {
                arma::vec z = x + (a/r)*p;
                return arma::dot(p,grad_f(z))/r;
            }
        );
        if (k%3==0 && alpha/r > -0.5) {
            k++;
            continue;
        } 
        x += (alpha/r)*p;
        k++;
    } while (r > tol);
    num_iter += k;
    exit_flag = 0;
}