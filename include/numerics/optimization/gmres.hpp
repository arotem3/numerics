#ifndef NUMERICS_OPTIMIZATION_GMRES_HPP
#define NUMERICS_OPTIMIZATION_GMRES_HPP

#include <cmath>
#include <vector>

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

namespace numerics
{
namespace optimization
{

namespace __gmres
{
template <typename real>
void givens_rotation(std::vector<real>& h, std::vector<real>& cs, std::vector<real>& sn, u_long k)
{
    for (u_long i=0; i < k; ++i)
    {
        real t = cs[i]*h[i] + sn[i]*h[i+1];
        h[i+1] = -sn[i]*h[i] + cs[i]*h[i+1];
        h[i] = t;
    }

    real t = std::sqrt( std::pow(h[k],2) + std::pow(h[k+1], 2) );
    cs.push_back(h[k] / t);
    sn.push_back(h[k+1] / t);

    h[k] = cs[k]*h[k] + sn[k]*h[k+1];
    h[k+1] = 0;
}

template<typename real>
bool solve_trimatu(std::vector<real>& y, const std::vector<std::vector<real>>& H, const std::vector<real>& beta)
{
    long k = H.size();
    
#if defined(NUMERICS_WITH_ARMA) && !defined(GMRES_NO_ARMA)
    arma::Col<real> b(k);
    for (long i=0; i < k; ++i)
        b[i] = beta[i];

    arma::Mat<real> U(k,k);
    for (long j=0; j < k; ++j)
        for (long i=0; i <= j; ++i)
            U.at(i,j) = H[j][i];

    bool success = arma::solve(b, arma::trimatu(U), b);
    if (success)
        y = arma::conv_to<std::vector<real>>::from(b);

    return success;
#else
    // back solve H*y = beta
    y = beta;
    if (H[k-1][k-1] == 0)
        return false;

    y[k-1] /= H[k-1][k-1];
    for (long i=k-2; i >= 0; --i)
    {
        for (long j=k-1; j >= i+1; --j)
            y[i] -= H[j][i] * y[j];

        if (H[i][i] == 0)
            return false;

        y[i] /= H[i][i];
    }

    return true;
#endif
}

template <class Vec, typename real=typename Vec::value_type>
bool solve_update(Vec& x, const std::vector<Vec>& Q, const std::vector<std::vector<real>>& H, const std::vector<real>& beta)
{
    long k = H.size();
    std::vector<real> y;
    bool success = solve_trimatu(y, H, beta);
    if (not success)
        return false;

    // orthogonal solve Q'*x = y
    for (u_long i=0; i < k; ++i)
        x += Q[i] * y[i];

    return true;
}
}

template <class Vec, class LinOp, class Dot, class Precond>
bool gmres(Vec& x, LinOp A, const Vec& b, Dot dot, Precond precond, typename Vec::value_type rtol, typename Vec::value_type atol, u_long restart, u_long max_cycles)
{
    typedef typename Vec::value_type real;
        
    bool success = false;

    real bnorm = std::sqrt( dot(b,b) );

    for (u_long i=0; i < max_cycles; ++i)
    {
        Vec r = b - A(x);
        r = precond(r);

        real rnorm = std::sqrt( dot(r,r) );
        real err = rnorm;

        std::vector<real> sn;
        std::vector<real> cs;
        std::vector<real> beta = {rnorm};

        std::vector<Vec> Q = {r/rnorm};
        std::vector<std::vector<real>> H;

        for (u_long k = 0; k < restart; ++k)
        {
            // arnoldi iteration
            Vec q = A(Q.back());
            q = precond(q);
            std::vector<real> h(k+2);
            for (u_long i=0; i <= k; ++i)
            {
                h[i] = dot(q, Q[i]);
                q -= h[i] * Q[i];
            }
            h[k+1] = std::sqrt( dot(q,q) );
            q /= h[k+1];

            Q.push_back(std::move(q));

            __gmres::givens_rotation(h, cs, sn, k);
            H.push_back(std::move(h));

            beta.push_back(-sn.at(k) * beta.at(k));
            beta.at(k) = cs.at(k) * beta.at(k);

            err = std::abs( beta.at(k+1) );
            if (err < rtol*bnorm + atol) {
                success = true;
                break;
            }
        }
        bool qr = __gmres::solve_update(x, Q, H, beta);
        success = success && qr;
        if (success or (not qr))
            break;
    }

    return success;
}

template <class Vec, class LinOp, class Dot>
inline bool gmres(Vec& x, LinOp A, const Vec& b, Dot dot, typename Vec::value_type rtol, typename Vec::value_type atol, u_long restart, u_long max_cycles)
{
    auto precond = [](const Vec& z) -> Vec
    {
        return z;
    };

    return gmres(x, std::forward<LinOp>(A), b, std::forward<Dot>(dot), precond, rtol, atol, restart, max_cycles);
}

#ifdef NUMERICS_WITH_ARMA
template <typename real, class Precond>
inline bool gmres(arma::Col<real>& x, const arma::Mat<real>& A, const arma::Col<real>& b, Precond precond, real rtol, real atol, u_long restart, u_long max_cycles)
{
    if (not A.is_square())
        throw std::invalid_argument("gmres() error: matrix A is not square.");
    if (A.n_rows != b.n_rows)
        throw std::invalid_argument("gmres() error: A.n_rows (=" + std::to_string(A.n_rows) + ") != b.n_rows (=" + std::to_string(A.n_rows) + ")");
    if (x.n_cols != A.n_cols)
        x = arma::zeros<arma::Col<real>>(A.n_cols);
    if (max_cycles == 0)
        max_cycles = x.n_elem;
    
    auto a = [&A](const arma::Col<real>& z) -> arma::Col<real>
    {
        return A*z;
    };

    return gmres(x, a, b, arma::dot<arma::Col<real>,arma::Col<real>>, std::forward<Precond>(precond), rtol, atol, restart, max_cycles);
}

template <typename real>
inline bool gmres(arma::Col<real>& x, const arma::Mat<real>& A, const arma::Col<real>& b, real rtol, real atol, u_long restart, u_long max_cycles)
{
    if (not A.is_square())
        throw std::invalid_argument("gmres() error: matrix A is not square.");
    if (A.n_rows != b.n_rows)
        throw std::invalid_argument("gmres() error: A.n_rows (=" + std::to_string(A.n_rows) + ") != b.n_rows (=" + std::to_string(A.n_rows) + ")");
    if (x.n_cols != A.n_cols)
        x = arma::zeros<arma::Col<real>>(A.n_cols);
    if (max_cycles == 0)
        max_cycles = x.n_elem;
    
    auto a = [&A](const arma::Col<real>& z) -> arma::Col<real>
    {
        return A*z;
    };

    auto precond = [](const arma::Col<real>& z) -> arma::Col<real>
    {
        return z;
    };
    return gmres(x, a, b, arma::dot<arma::Col<real>,arma::Col<real>>, precond, rtol, atol, restart, max_cycles);
}

template <typename real, class Precond>
inline bool gmres(arma::Col<real>& x, const arma::SpMat<real>& A, const arma::Col<real>& b, Precond precond, real rtol, real atol, u_long restart, u_long max_cycles)
{
    if (not A.is_square())
        throw std::invalid_argument("gmres() error: matrix A is not square.");
    if (A.n_rows != b.n_rows)
        throw std::invalid_argument("gmres() error: A.n_rows (=" + std::to_string(A.n_rows) + ") != b.n_rows (=" + std::to_string(A.n_rows) + ")");
    if (x.n_cols != A.n_cols)
        x = arma::zeros<arma::Col<real>>(A.n_cols);
    if (max_cycles == 0)
        max_cycles = x.n_elem;

    auto a = [&A](const arma::Col<real>& z) -> arma::Col<real>
    {
        return A*z;
    };
    return gmres(x, a, b, arma::dot<arma::Col<real>,arma::Col<real>>, std::forward<Precond>(precond), rtol, atol, restart, max_cycles);
}

template<typename real>
inline bool gmres(arma::Col<real>& x, const arma::SpMat<real>& A, const arma::Col<real>& b, real rtol, real atol, u_long restart, u_long max_cycles)
{
    if (not A.is_square())
        throw std::invalid_argument("gmres() error: matrix A is not square.");
    if (A.n_rows != b.n_rows)
        throw std::invalid_argument("gmres() error: A.n_rows (=" + std::to_string(A.n_rows) + ") != b.n_rows (=" + std::to_string(A.n_rows) + ")");
    if (x.n_cols != A.n_cols)
        x = arma::zeros<arma::Col<real>>(A.n_cols);
    if (max_cycles == 0)
        max_cycles = x.n_elem;
    
    
    auto a = [&A](const arma::Col<real>& z) -> arma::Col<real>
    {
        return A*z;
    };

    auto precond = [](const arma::Col<real>& z) -> arma::Col<real>
    {
        return z;
    };
    return gmres(x, a, b, arma::dot<arma::Col<real>,arma::Col<real>>, precond, rtol, atol, restart, max_cycles);
}
#endif
}
}
#endif