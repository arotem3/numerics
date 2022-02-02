#ifndef NUMERICS_OPTIMIZATION_SIMPLEX_HPP
#define NUMERICS_OPTIMIZATION_SIMPLEX_HPP

#include "numerics/optimization/optim_base.hpp"

namespace numerics {
    namespace optimization {
        #ifdef NUMERICS_WITH_ARMA
        // solves the linear program: min dot(f,x) such that A*x <= b and x >= 0 using
        // the simplex method.
        // see:
        // https://en.wikipedia.org/wiki/Simplex_algorithm
        template <std::floating_point real>
        real simplex(arma::Col<real>& x, const arma::Col<real>& f, const arma::Mat<real>& A, const arma::Col<real>& b) {
            u_long n = A.n_rows, m = A.n_cols;

            /* S =  [  A        I           b]
                    [ -f    (0 0 ... 1)     0] */

            arma::Mat<real> S = arma::zeros<arma::Mat<real>>(n + 1, m + n + 2);
            S(arma::span(0, n-1), arma::span(0, m-1)) = A;
            S(n, arma::span(0, m-1)) = -f.as_row();
            for (u_long i=0; i < n+1; ++i)
                S(i, m+i) = 1;
            S(arma::span(0,n-1), m+n+1) = b;

            u_long nr = S.n_rows;
            u_long nc = S.n_cols;

            while ( not arma::all(S.row(nr-1) >= 0) )
            {
                u_long pivCol = arma::index_min( S.row(nr-1) ); // min of last row
                u_long pivRow = arma::index_min( S(arma::span(0,nr-2), nc-1) / S(arma::span(0,nr-2),pivCol) ); // min of (piv col)/(end col);
                
                S.row(pivRow) /= S(pivRow,pivCol);
                for (u_long i=0; i < nr; ++i)
                    if (i != pivRow)
                        S.row(i) -= S.row(pivRow) * S(i,pivCol); // row reduce set all non-pivot positions in pivot col to zero
            }
            arma::urowvec nonbas = (S(nr-1, arma::span(0,nc-2)) == 0); // non-basic variables
            arma::Mat<real> B = S(arma::span(0,nr-2), arma::span(0,nc-2)); // constraint matrix with out LHS
            for (u_long i=0; i < B.n_cols; ++i)
                B.col(i) *= nonbas(i);
            
            // B = B.cols(arma::find(nonbas));

            x = arma::solve(B, S(arma::span(0,nr-2), nc-1)); // solve B\LHS
            x = x(arma::span(0, nc-nr-2)); //return where min occurs

            return arma::dot(f,x);
        }
        #endif
    } // namespace optimization
} // namespace numerics

#endif