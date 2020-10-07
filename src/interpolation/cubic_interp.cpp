#include "numerics.hpp"

numerics::PieceWisePoly numerics::natural_cubic_spline(const arma::vec& x, const arma::vec&y, const std::string& extrapolation, double val) {
    PieceWisePoly out(extrapolation,val);
    
    out._check_xy(x, y);

    u_long n = x.n_elem - 1;

    arma::vec h = arma::zeros(n);
    arma::sp_mat A(n+1,n+1);
    arma::vec RHS = arma::zeros(n+1);
    arma::vec b = arma::zeros(n);
    arma::vec d = arma::zeros(n);
    arma::uvec I = arma::sort_index(x);
    out._x = x(I);
    out._check_x(out._x);
    arma::vec _y = y(I);

    for (u_long i=1; i < n+1; ++i) {
        h(i-1) = out._x(i) - out._x(i-1); 
    }

    arma::vec subD = h;
    arma::vec supD(arma::size(subD),arma::fill::zeros);
    arma::vec mainD(n+1,arma::fill::zeros);

    subD(n-1) = 0;
    mainD(0) = 1;
    mainD(n) = 1;
    supD(0) = 0;

    for (u_long i=1; i < n; ++i) {     
        mainD(i) = 2 * (h(i) + h(i-1));
        supD(i) = h(i);

        RHS(i) = 3 * (_y(i+1) - _y(i))/h(i) - 3 * (_y(i) - _y(i-1))/h(i-1);
    }

    A.diag(-1) = subD;
    A.diag()   = mainD;
    A.diag(1)  = supD;

    arma::vec c = spsolve(A,RHS);

    for (u_long i=0; i < n; ++i) {
        b(i) = (_y(i+1) - _y(i))/h(i) - h(i)*(2*c(i) + c(i+1))/3;
        d(i) = (c(i+1) - c(i))/(3*h(i));
    }
    c = c.rows(0,n-1);

    for (u_long i=0; i < n; ++i) {
        arma::vec p = {
            d(i),
            c(i) - 3*d(i)*out._x(i),
            b(i) - 2*c(i)*out._x(i) + 3*d(i)*std::pow(out._x(i),2),
            _y(i) - b(i)*out._x(i) + c(i)*std::pow(out._x(i),2) - d(i)*std::pow(out._x(i),3)
        }; // convert spline to polynomial
        out._P.push_back(Polynomial(p));
    }
    return out;
}

// numerics::CubicInterp::CubicInterp(const arma::vec& x, const arma::vec& y, const std::string& extrapolation, double val) : PieceWisePoly(extrapolation,val) {
//     _check_xy(x, y);

//     u_long n = x.n_elem - 1;

//     arma::vec h = arma::zeros(n);
//     arma::sp_mat A(n+1,n+1);
//     arma::vec RHS = arma::zeros(n+1);
//     arma::vec b = arma::zeros(n);
//     arma::vec d = arma::zeros(n);
//     arma::uvec I = arma::sort_index(x);
//     _x = x(I);
//     _check_x(_x);
//     arma::vec _y = y(I);

//     for (u_long i=1; i < n+1; ++i) {
//         h(i-1) = _x(i) - _x(i-1); 
//     }

//     arma::vec subD = h;
//     arma::vec supD(arma::size(subD),arma::fill::zeros);
//     arma::vec mainD(n+1,arma::fill::zeros);

//     subD(n-1) = 0;
//     mainD(0) = 1;
//     mainD(n) = 1;
//     supD(0) = 0;

//     for (u_long i=1; i < n; ++i) {     
//         mainD(i) = 2 * (h(i) + h(i-1));
//         supD(i) = h(i);

//         RHS(i) = 3 * (_y(i+1) - _y(i))/h(i) - 3 * (_y(i) - _y(i-1))/h(i-1);
//     }

//     A.diag(-1) = subD;
//     A.diag()   = mainD;
//     A.diag(1)  = supD;

//     arma::vec c = spsolve(A,RHS);

//     for (u_long i=0; i < n; ++i) {
//         b(i) = (_y(i+1) - _y(i))/h(i) - h(i)*(2*c(i) + c(i+1))/3;
//         d(i) = (c(i+1) - c(i))/(3*h(i));
//     }
//     c = c.rows(0,n-1);

//     for (u_long i=0; i < n; ++i) {
//         arma::vec p = {
//             d(i),
//             c(i) - 3*d(i)*_x(i),
//             b(i) - 2*c(i)*_x(i) + 3*d(i)*std::pow(_x(i),2),
//             _y(i) - b(i)*_x(i) + c(i)*std::pow(_x(i),2) - d(i)*std::pow(_x(i),3)
//         }; // convert spline to polynomial
//         _P.push_back(Polynomial(p));
//     }
// }