#include <numerics.hpp>

/* init_simplex(x) : initialize a simplex from an intial guess using randomized orthonormal directions.
 * --- x : initial guess of optimal point */
arma::mat numerics::optimization::NelderMead::_init_simplex(const arma::vec& x) {
    int n = x.n_elem;
    arma::mat dx = arma::randn(n,1);
    dx = _side_length*arma::join_rows( arma::orth(dx), arma::null(dx.t()) ); // spanning orthonormal matrix
    dx.each_col() += x;
    return arma::join_rows(x,dx);
}

void numerics::optimization::NelderMead::minimize(arma::vec& x, const dFunc& f) {
    int m=x.n_elem;
    arma::mat xx = _init_simplex(x);
    arma::vec yy = arma::zeros(xx.n_cols);
    for (int i=0; i < m+1; ++i) {
        yy(i) = f(xx.col(i));
    }

    int worst, scndw, best;
    _n_iter = 0;
    VerboseTracker T(_max_iter);
    if (_v) T.header();
    while (true) {
        arma::uvec ind = arma::sort_index(yy);
        worst = ind(m);
        scndw = ind(m-1);
        best  = ind(0);

        // reflect x(worst) accross center
        arma::vec c = arma::mean(xx.cols(ind.rows(0,m-1)),1);
        arma::vec xr = c + _step * (c - xx.col(worst));
        double yr = f(xr);
        if (yy(best) < yr && yr < yy(scndw)) {
            xx.col(worst) = xr;
            yy(worst) = yr;
        } else if (yr < yy(best)) { // new point is very good, attempt further search in this direction
            arma::vec xe = c + _expand * (xr - c);
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
                arma::vec xc = c + _contract * (xr - c);
                double yc = f(xc);
                if (yc < yr) {
                    xx.col(worst) = xc;
                    yy(worst) = yc;
                } else { // shrink simplex
                    for (int i=0; i < m+1; ++i) {
                        if (i==best) continue;
                        xx.col(i) = xx.col(best) + _shrink * (xx.col(i) - xx.col(best));
                        yy(i) = f(xx.col(i));
                    }
                }
            } else if (yr > yy(worst)) { // contraction inside simplex
                arma::vec xc = c + _contract * (xx.col(worst) - c);
                double yc = f(xc);
                if (yc < yy(worst)) {
                    xx.col(worst) = xc;
                    yy(worst) = yc;
                } else { // shrink simplex
                    for (int i=0; i < m+1; ++i) {
                        if (i==best) continue;
                        xx.col(i) = xx.col(best) + _shrink * (xx.col(i) - xx.col(best));
                        yy(i) = f(xx.col(i));
                    }
                }
            }
        }
        if (_v) T.iter(_n_iter, yy(best));
        _n_iter++;

        double ftol = _ftol*std::max(1.0, std::abs<double>(yy(best)));
        if (std::abs(yy(best) - yy(worst)) < ftol) {
            _exit_flag = 0;
            if (_v) T.success_flag();
            break;
        }

        double xtol = _xtol*std::max(1.0, arma::norm(xx.col(best)));
        if (arma::norm(xx.col(worst) - xx.col(best),"inf") > xtol) {
            _exit_flag = 1;
            if (_v) T.success_flag();
            break;
        }

        if (_n_iter >= _max_iter) {
            _exit_flag = 2;
            if (_v) T.max_iter_flag();
            break;
        }


    }
    u_int i = yy.index_min();
    x = xx.col(i);
}