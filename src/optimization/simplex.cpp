#include "numerics.hpp"

double numerics::optimization::simplex(arma::vec& x, const arma::vec& f, const arma::mat& conRHS, const arma::vec& conLHS) {
    u_long numCons = conRHS.n_elem;
    arma::mat A = arma::join_cols(conRHS, -f.as_row()); // A = [RHS; f]
    A = arma::join_rows(A, arma::eye(A.n_rows,A.n_rows)); // A = [A , I]
    arma::vec z = {0};
    arma::vec LHS = arma::join_cols(conLHS,z); // LHS = [LHS; 0]
    A = arma::join_rows(A,LHS); // A = [A, LHS]
    
    u_long numRows = A.n_rows;
    u_long numCols = A.n_cols;

    while ( not arma::all(A.row(numRows-1) >= 0) ) {
        u_long pivCol = arma::index_min( A.row(numRows-1) ); // min of last row
        u_long pivRow = arma::index_min( A(arma::span(0,numRows-2), numCols-1) / A(arma::span(0,numRows-2),pivCol) ); // min of (piv col)/(end col);
        A.row(pivRow) /= A(pivRow,pivCol);
        for (u_long i(0); i < numRows; ++i) {
            if (i != pivRow) {
                A.row(i) -= A.row(pivRow) * A(i,pivCol); // row reduce set all non-pivot positions in pivot col to zero
            }
        }
    }
    arma::urowvec nonbas = (A(numRows-1, arma::span(0,numCols-2)) == 0); // non-basic variables
    arma::mat B = A(arma::span(0,numRows-2), arma::span(0,numCols-2)); // constraint matrix with out LHS
    for (u_long i(0); i <= numRows-2; ++i) {
        B.col(i) *= nonbas(i);
    }
    x = solve(B, A(arma::span(0,numRows-2), numCols-1)); // solve B\LHS
    x = x(arma::span(0, numCols-numRows-2)); //return where min occurs
    return arma::dot(f,x);
}