#ifndef NUMERICS_OPTIMIZATION_PCG_HPP
#define NUMERICS_OPTIMIZATION_PCG_HPP

#include <cmath>

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

namespace numerics
{
namespace optimization
{
// pcg() solves the linear equation A*x == b where A is symmetric positive
// definite. This implementation is matrix free and requires A take in a Vec
// object and return a Vec object. Similarly the dot object should take two vec
// objects and return a scalar. The Vec class should act like a mathematical
// vector with the operators (+,-,+=,-=) defined between two Vec instances and
// (*) between scalars and Vec. arma::Mat and std::valarray both satisfy these
// conditions. The Precond object takes in a Vec object and returns a Vec
// object, if it is not defined, then the identity preconditioner will be used
// (i.e. no preconditioner). For armadillo types, the function is specialized so
// that the dot object is not necessary.

template<class Vec, class LinOp, class Dot, class Precond, typename real=typename Vec::value_type>
bool pcg(Vec& x, LinOp A, const Vec& b, Dot dot, Precond precond, real rtol, real atol, u_long max_iter)
{
    Vec r = b - A(x);
    real bnorm = std::sqrt(dot(b,b));

    bool success = false;

    Vec p, z;
    real rho_prev, rho;
    for (u_long i=0; i < max_iter; ++i)
    {
        z = precond(r);
        rho = dot(r, z);

        if (i == 0)
            p = z;
        else {
            real beta = rho / rho_prev;
            p = z + beta * p;
        }
        
        Vec q = A(p);
        real alpha = rho / dot(p, q);
        x += alpha * p;
        r -= alpha * q;

        rho_prev = rho;

        if (std::sqrt(dot(r,r)) < rtol*bnorm + atol)
        {
            success = true;
            break;
        }
    }

    return success;
}

template<class Vec, class LinOp, class Dot, typename real=typename Vec::value_type>
inline bool pcg(Vec& x, LinOp A, const Vec& b, Dot dot, real rtol, real atol, u_long max_iter)
{
    auto precond = [](const Vec& z) -> Vec
    {
        return z;
    };
    return pcg(x, std::forward<LinOp>(A), b, std::forward<Dot>(dot), precond, rtol, atol, max_iter);
}

#ifdef NUMERICS_WITH_ARMA
template<typename real>
inline bool pcg(arma::Col<real>& x, const arma::Mat<real>& A, const arma::Col<real>& b, real rtol, real atol, u_long max_iter)
{
    if (not A.is_sympd())
        throw std::invalid_argument("pcg() error: matrix A is not symmetric positive definite.");
    if (A.n_rows != b.n_rows)
        throw std::invalid_argument("pcg() error: A.n_rows (=" + std::to_string(A.n_rows) + ") != b.n_rows (=" + std::to_string(b.n_rows) + ")");
    if (x.n_rows != A.n_cols)
        x = arma::zeros<arma::Col<real>>(A.n_cols);
    if (max_iter == 0)
        max_iter = x.n_elem;

    auto lin_op = [&A](const arma::Col<real>& z) -> arma::Col<real>
    {
        return A*z;
    };
    auto precond = [](const arma::Col<real>& z) -> arma::Col<real>
    {
        return z;
    };
    return pcg(x, lin_op, b, arma::dot<arma::Col<real>,arma::Col<real>>, precond, rtol, atol, max_iter);
}

template<typename real, class Precond>
inline bool pcg(arma::Col<real>& x, const arma::Mat<real>& A,   const arma::Col<real>& b, Precond precond, real rtol, real atol, u_long max_iter)
{
    if (not A.is_sympd())
        throw std::invalid_argument("pcg() error: matrix A is not symmetric positive definite.");
    if (A.n_rows != b.n_rows)
        throw std::invalid_argument("pcg() error: A.n_rows (=" + std::to_string(A.n_rows) + ") != b.n_rows (=" + std::to_string(A.n_rows) + ")");
    if (x.n_cols != A.n_cols)
        x = arma::zeros<arma::Col<real>>(A.n_cols);
    if (max_iter == 0)
        max_iter = x.n_elem;
    
    auto lin_op = [&A](const arma::Col<real>& z) -> arma::Col<real>
    {
        return A*z;
    };
    return pcg(x, lin_op, b, arma::dot<arma::Col<real>,arma::Col<real>>, std::forward<Precond>(precond), rtol, atol, max_iter);
}

template<typename real>
inline bool pcg(arma::Col<real>& x, const arma::SpMat<real>& A, const arma::Col<real>& b, real rtol, real atol, u_long max_iter)
{
    if (not A.is_symmetric())
        throw std::invalid_argument("pcg() error: matrix A is not symmetric positive definite.");
    if (A.n_rows != b.n_rows)
        throw std::invalid_argument("pcg() error: A.n_rows (=" + std::to_string(A.n_rows) + ") != b.n_rows (=" + std::to_string(A.n_rows) + ")");
    if (x.n_cols != A.n_cols)
        x = arma::zeros<arma::Col<real>>(A.n_cols);
    if (max_iter == 0)
        max_iter = x.n_elem;
    
    auto lin_op = [&A](const arma::Col<real>& z) -> arma::Col<real>
    {
        return A*z;
    };
    auto precond = [](const arma::Col<real>& z) -> arma::Col<real>
    {
        return z;
    };
    return pcg(x, lin_op, b, arma::dot<arma::Col<real>,arma::Col<real>>, precond, rtol, atol, max_iter);
}

template<typename real, class Precond>
inline bool pcg(arma::Col<real>& x, const arma::SpMat<real>& A, const arma::Col<real>& b, Precond precond, real rtol, real atol, u_long max_iter)
{
    if (not A.is_symmetric())
        throw std::invalid_argument("pcg() error: matrix A is not symmetric positive definite.");
    if (A.n_rows != b.n_rows)
        throw std::invalid_argument("pcg() error: A.n_rows (=" + std::to_string(A.n_rows) + ") != b.n_rows (=" + std::to_string(A.n_rows) + ")");
    if (x.n_cols != A.n_cols)
        x = arma::zeros<arma::Col<real>>(A.n_cols);
    if (max_iter == 0)
        max_iter = x.n_elem;
    
    auto lin_op = [&A](const arma::Col<real>& z) -> arma::Col<real>
    {
        return A*z;
    };
    return pcg(x, lin_op, b, arma::dot<arma::Col<real>,arma::Col<real>>, std::forward<Precond>(precond), rtol, atol, max_iter);
}
#endif

}
}

#endif