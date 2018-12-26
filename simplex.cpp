#include "numerics.hpp"

//--- simplex method for solving linear constrained maximization problems ---//
//----- A  : simplex matrix predefined --------------------------------------//
//----- x  : where solution goes --------------------------------------------//
//----- returns minimum value -----------------------------------------------//
double numerics::simplex(arma::mat& A, arma::vec& x) {
    int numRows = A.n_rows;
    int numCols = A.n_cols;
    arma::mat D = A;
    auto f = [&D,numRows,numCols](arma::vec x) -> double { // our objective function
        arma::rowvec b = D(numRows-1, arma::span(0,numCols-numRows-2));
        return arma::dot(-b.t(), x);
    };

    short k = 0;
    while ( !arma::all(A.row(numRows-1) >= 0) ) {
        if (k > 100) { //simplex takes too long
            std::cerr << "simplex() failed: too many iterations needed to reduce system." << std::endl;
            x = {NAN};
            return NAN;
        }
        int pivCol = arma::index_min( A.row(numRows-1) ); // min of last row
        int pivRow = arma::index_min( A(arma::span(0,numRows-2), numCols-1) / A(arma::span(0,numRows-2),pivCol) ); // min of (piv col)/(end col);
        A.row(pivRow) /= A(pivRow,pivCol);
        for (int i(0); i < numRows; ++i) {
            if (i != pivRow) {
                A.row(i) -= A.row(pivRow) * A(i,pivCol); // row reduce set all non-pivot positions in pivot col to zero
            }
        }
        k++;
    }
    arma::urowvec nonbas = (A(numRows-1, arma::span(0,numCols-2)) == 0); // non-basic variables
    arma::mat B = A(arma::span(0,numRows-2), arma::span(0,numCols-2)); // constraint matrix with out LHS
    for (int i(0); i <= numRows-2; ++i) {
        B.col(i) *= nonbas(i);
    }
    x = solve(B, A(arma::span(0,numRows-2), numCols-1)); // solve B\LHS
    x = x(arma::span(0, numCols-numRows-2)); //return where min occurs
    A = D;
    return f(x);
}

//--- simplex method overload for those who would rather not pre-format ---//
//----- f  : z(x) = f*x = dot(f,x); function to maximize ------------------//
//----- conRHS : right hand side of constraint equations ------------------//
//----- conLHS : left hand side of constraint equations -------------------//
//---------- conRHS*x <= conLHS -------------------------------------------//
//----- x  : where solution is stored -------------------------------------//
double numerics::simplex(const arma::rowvec& f, const arma::mat& conRHS, const arma::vec& conLHS, arma::vec& x) {
    int numCons = conRHS.n_elem;
    arma::mat A = arma::join_cols(conRHS, -f); // A = [RHS; f]
    A = arma::join_rows(A, arma::eye(numCons+1,numCons+1)); // A = [A , I]
    arma::vec z = {0};
    arma::vec LHS = arma::join_cols(conLHS,z); // LHS = [LHS; 0]
    A = arma::join_rows(A,LHS); // A = [A, LHS]
    return simplex(A,x);
}