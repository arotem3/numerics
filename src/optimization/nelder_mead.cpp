#include <numerics.hpp>

/* init_simplex(x) : initialize a simplex from an intial guess using randomized orthonormal directions.
 * --- x : initial guess of optimal point */
arma::mat numerics::nelder_mead::init_simplex(const arma::vec& x) {
    int n = x.n_elem;
    arma::mat dx = arma::randn(n,1);
    dx = initial_side_length*arma::join_rows( arma::orth(dx), arma::null(dx.t()) ); // spanning orthonormal matrix
    dx.each_col() += x;
    return arma::join_rows(x,dx);
}

/* minimize(f, x) : minimize a multivariate function using the Nelder-Mead algorithm.
 * --- f : f(x) function to minimize.
 * --- x : initial guess of minimum point, solution point will be assigned here. */
void numerics::nelder_mead::minimize(const std::function<double(const arma::vec&)>& f, arma::vec& x) {
    int m=x.n_elem;
    arma::mat xx = init_simplex(x);
    arma::vec yy = arma::zeros(xx.n_cols);
    for (int i=0; i < m+1; ++i) {
        yy(i) = f(xx.col(i));
    }

    int worst, scndw, best;
    int k = 0;
    do {
        arma::uvec ind = arma::sort_index(yy);
        worst = ind(m);
        scndw = ind(m-1);
        best  = ind(0);

        if (k >= max_iterations) {
            exit_flag = 1;
            num_iter += k;
            x = xx.col(best);
            return;
        }

        // reflect x(worst) accross center
        arma::vec c = arma::mean(xx.cols(ind.rows(0,m-1)),1);
        arma::vec xr = c + step * (c - xx.col(worst));
        double yr = f(xr);
        if (yy(best) < yr && yr < yy(scndw)) {
            xx.col(worst) = xr;
            yy(worst) = yr;
        } else if (yr < yy(best)) { // new point is very good, attempt further search in this direction
            arma::vec xe = c + expand * (xr - c);
            double ye = f(xe);
            if (ye < yr) {
                xx.col(worst) = xe;
                yy(worst) = ye;
            } else {
                xx.col(worst) = xr;
                yy(worst) = yr;
            }
        } else if (yy(scndw) < yr) { // potential over shoot
            if (yy(scndw) < yr && yr < yy(worst)) { // contraction outside simplex
                arma::vec xc = c + contract * (xr - c);
                double yc = f(xc);
                if (yc < yr) {
                    xx.col(worst) = xc;
                    yy(worst) = yc;
                } else { // shrink simplex
                    for (int i=0; i < m+1; ++i) {
                        if (i==best) continue;
                        xx.col(i) = xx.col(best) + shrink * (xx.col(i) - xx.col(best));
                        yy(i) = f(xx.col(i));
                    }
                }
            } else if (yr > yy(worst)) { // contraction inside simplex
                arma::vec xc = c + contract * (xx.col(worst) - c);
                double yc = f(xc);
                if (yc < yy(worst)) {
                    xx.col(worst) = xc;
                    yy(worst) = yc;
                } else { // shrink simplex
                    for (int i=0; i < m+1; ++i) {
                        if (i==best) continue;
                        xx.col(i) = xx.col(best) + shrink * (xx.col(i) - xx.col(best));
                        yy(i) = f(xx.col(i));
                    }
                }
            }
        }
        k++;
    } while (arma::norm(xx.col(scndw) - xx.col(best),"inf") > tol);
    int i = yy.index_min();
    x = xx.col(i);
    exit_flag = 0;
    num_iter += k;
}