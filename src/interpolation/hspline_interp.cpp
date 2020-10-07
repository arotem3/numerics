#include <numerics.hpp>

numerics::PieceWisePoly numerics::hermite_cubic_spline(const arma::vec& x, const arma::vec& y, const std::string& extrapolation, double val) {
    PieceWisePoly out(extrapolation,val);
    
    out._check_xy(x,y);

    u_long n = x.n_elem;
    arma::uvec I = arma::sort_index(x);
    out._x = x(I);
    out._check_x(out._x);
    arma::vec _y = y(I);

    arma::sp_mat D;
    ode::diffmat(D,out._x);
    arma::vec _dy = D*_y;

    arma::vec h = out._x.rows(1,n-1) - out._x.rows(0,n-2);
    arma::vec d = (2*(_y.rows(0,n-2) - _y.rows(1,n-1)) + (_dy.rows(0,n-2) + _dy.rows(1,n-1))%h) / arma::pow(h,3);
    arma::vec c = -(3*(_y.rows(0,n-2) - _y.rows(1,n-1)) + (2*_dy.rows(0,n-2) + _dy.rows(1,n-1))%h) / arma::pow(h,2);

    for (u_int i=0; i < n-1; ++i) {
        arma::vec p = {
            d(i),
            c(i) - 3*d(i)*out._x(i),
            _dy(i) - 2*c(i)*out._x(i) + 3*d(i)*std::pow(out._x(i),2),
            _y(i) - _dy(i)*out._x(i) + c(i)*std::pow(out._x(i),2) - d(i)*std::pow(out._x(i),3)
        };
        out._P.push_back(Polynomial(p));
    }
    return out;
}

numerics::PieceWisePoly numerics::hermite_cubic_spline(const arma::vec& x, const arma::vec& y, const arma::vec& yp, const std::string& extrapolation, double val) {
    PieceWisePoly out(extrapolation,val);
    out._check_xy(x,y);
    out._check_xy(x,yp);
    
    u_long n = x.n_elem;
    arma::uvec I = arma::sort_index(x);
    out._x = x(I);
    out._check_x(out._x);
    arma::vec _y = y(I);
    arma::vec _dy = yp(I);

    arma::vec h = out._x.rows(1,n-1) - out._x.rows(0,n-2);
    arma::vec d = (2*(_y.rows(0,n-2) - _y.rows(1,n-1)) + (_dy.rows(0,n-2) + _dy.rows(1,n-1))%h) / arma::pow(h,3);
    arma::vec c = -(3*(_y.rows(0,n-2) - _y.rows(1,n-1)) + (2*_dy.rows(0,n-2) + _dy.rows(1,n-1))%h) / arma::pow(h,2);

    for (u_int i=0; i < n-1; ++i) {
        arma::vec p = {
            d(i),
            c(i) - 3*d(i)*out._x(i),
            _dy(i) - 2*c(i)*out._x(i) + 3*d(i)*std::pow(out._x(i),2),
            _y(i) - _dy(i)*out._x(i) + c(i)*std::pow(out._x(i),2) - d(i)*std::pow(out._x(i),3)
        };
        out._P.push_back(Polynomial(p));
    }
    return out;
}

// numerics::HSplineInterp::HSplineInterp(const arma::vec& x, const arma::vec& y, const arma::vec& yp, const std::string& extrapolation, double val) : PieceWisePoly(extrapolation,val) {
//     _check_xy(x,y);
//     _check_xy(x,yp);
    
//     u_long n = x.n_elem;
//     arma::uvec I = arma::sort_index(x);
//     _x = x(I);
//     _check_x(_x);
//     arma::vec _y = y(I);
//     arma::vec _dy = yp(I);

//     arma::vec h = _x.rows(1,n-1) - _x.rows(0,n-2);
//     arma::vec d = (2*(_y.rows(0,n-2) - _y.rows(1,n-1)) + (_dy.rows(0,n-2) + _dy.rows(1,n-1))%h) / arma::pow(h,3);
//     arma::vec c = -(3*(_y.rows(0,n-2) - _y.rows(1,n-1)) + (2*_dy.rows(0,n-2) + _dy.rows(1,n-1))%h) / arma::pow(h,2);

//     for (u_int i=0; i < n-1; ++i) {
//         arma::vec p = {
//             d(i),
//             c(i) - 3*d(i)*_x(i),
//             _dy(i) - 2*c(i)*_x(i) + 3*d(i)*std::pow(x(i),2),
//             _y(i) - _dy(i)*_x(i) + c(i)*std::pow(x(i),2) - d(i)*std::pow(x(i),3)
//         };
//         _P.push_back(Polynomial(p));
//     }
// }

// numerics::HSplineInterp::HSplineInterp(const arma::vec& x, const arma::vec& y, const std::string& extrapolation, double val) : PieceWisePoly(extrapolation,val) {
//     _check_xy(x,y);

//     u_long n = x.n_elem;
//     arma::uvec I = arma::sort_index(x);
//     _x = x(I);
//     _check_x(_x);
//     arma::vec _y = y(I);

//     arma::sp_mat D;
//     ode::diffmat(D,_x);
//     arma::vec _dy = D*_y;

//     arma::vec h = _x.rows(1,n-1) - _x.rows(0,n-2);
//     arma::vec d = (2*(_y.rows(0,n-2) - _y.rows(1,n-1)) + (_dy.rows(0,n-2) + _dy.rows(1,n-1))%h) / arma::pow(h,3);
//     arma::vec c = -(3*(_y.rows(0,n-2) - _y.rows(1,n-1)) + (2*_dy.rows(0,n-2) + _dy.rows(1,n-1))%h) / arma::pow(h,2);

//     for (u_int i=0; i < n-1; ++i) {
//         arma::vec p = {
//             d(i),
//             c(i) - 3*d(i)*_x(i),
//             _dy(i) - 2*c(i)*_x(i) + 3*d(i)*std::pow(x(i),2),
//             _y(i) - _dy(i)*_x(i) + c(i)*std::pow(x(i),2) - d(i)*std::pow(x(i),3)
//         };
//         _P.push_back(Polynomial(p));
//     }
// }
