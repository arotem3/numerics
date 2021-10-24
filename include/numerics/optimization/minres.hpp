#ifndef NUMERICS_OPTIMIZATION_MINRES_HPP
#define NUMERICS_OPTIMIZATION_MINRES_HPP

#include <limits>
#include <cmath>

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

// conversion of the matlab code: https://web.stanford.edu/group/SOL/software/minres/
// which is based off of:
// C. C. Paige and M. A. Saunders (1975). Solution of sparse indefinite systems of linear equations, SIAM J. Numerical Analysis 12, 617-629.

namespace numerics
{
namespace optimization
{

template <class Vec, class LinOp, class Dot, class Precond, typename real=typename Vec::value_type>
bool minres(Vec& x, LinOp A, const Vec& b, Dot dot, Precond precond, real rtol, real atol, u_long maxit)
{
    bool success = false;
    
    Vec r1 = b - A(x);
    Vec y = precond(r1);
    Vec r2 = r1;

    real machine_epsilon = std::numeric_limits<real>::epsilon();
    real beta1 = std::sqrt( dot(y, r1) );
    real beta = beta1, beta_old = 0.0f, tnorm = 0.0f, dbar = 0.0f,
        eps = 0.0f, eps_old = 0.0f, phibar = beta1, cs = -1.0f, sn = 0.0f;

    Vec w = 0*x;
    Vec w2 = w;

    for (u_long i=1; i <= maxit; ++i)
    {
        Vec v = y / beta;
        y = A(v);

        if (i >= 2)
            y -= (beta/beta_old) * r1;

        real alpha = dot(v, y);
        y -= (alpha/beta) * r2;

        r1 = std::move(r2);
        r2 = y;

        y = precond(r2);

        beta_old = beta;
        beta = std::sqrt( dot(y,y) );

        tnorm += alpha*alpha + beta*beta + beta_old*beta_old;

        eps_old = eps;
        real delta = cs*dbar + sn*alpha;
        real gbar = sn*dbar - cs*alpha;

        eps = sn*beta;
        dbar = -cs*beta;

        real root = std::sqrt(gbar*gbar + dbar*dbar);
        real gamma = std::max(std::sqrt(gbar*gbar + beta*beta), machine_epsilon);
        cs = gbar / gamma;
        sn = beta / gamma;

        real phi = cs * phibar;
        phibar *= sn;

        Vec w1 = std::move(w2);
        w2 = std::move(w);

        w = (v - eps_old*w1 - delta*w2)/gamma;
        x += phi*w;

        real A_norm = std::sqrt(tnorm);
        real y_norm = std::sqrt( dot(x,x) );
        
        bool residual_convergence = phibar < rtol*(A_norm * y_norm) + atol;
        bool residual_orthogonality = root < rtol*A_norm + atol;
        if (residual_convergence or residual_orthogonality) {
            success = true;
            break;
        }
    }

    return success;
}

template <class Vec, class LinOp, class Dot, typename real=typename Vec::value_type>
inline bool minres(Vec& x, LinOp A, const Vec& b, Dot dot, real rtol, real atol, u_long maxit)
{
    auto precond = [](const Vec& z) -> Vec
    {
        return z;
    };

    return minres(x, std::forward<LinOp>(A), b, std::forward<Dot>(dot), precond, rtol, atol, maxit);
}

#ifdef NUMERICS_WITH_ARMA
template <typename real, class Precond>
inline bool minres(arma::Col<real>& x, const arma::Mat<real>& A, const arma::Col<real>& b, Precond precond, real rtol, real atol, u_long maxit)
{
    auto a = [&A](const arma::Col<real>& x) -> arma::Col<real>
    {
        return A*x;
    };

    return minres(x, a, b, arma::dot<arma::Col<real>,arma::Col<real>>, std::forward<Precond>(precond), rtol, atol, maxit);
}

template <typename real>
inline bool minres(arma::Col<real>& x, const arma::Mat<real>& A, const arma::Col<real>& b, real rtol, real atol, u_long maxit)
{
    auto a = [&A](const arma::Col<real>& x) -> arma::Col<real>
    {
        return A*x;
    };

    auto precond = [](const arma::Col<real>& z) -> arma::Col<real>
    {
        return z;
    };

    return minres(x, a, b, arma::dot<arma::Col<real>,arma::Col<real>>, precond, rtol, atol, maxit);
}

template <typename real, class Precond>
inline bool minres(arma::Col<real>& x, const arma::SpMat<real>& A, const arma::Col<real>& b, Precond precond, real rtol, real atol, u_long maxit)
{
    auto a = [&A](const arma::Col<real>& x) -> arma::Col<real>
    {
        return A*x;
    };

    return minres(x, a, b, arma::dot<arma::Col<real>,arma::Col<real>>, std::forward<Precond>(precond), rtol, atol, maxit);
}

template <typename real>
inline bool minres(arma::Col<real>& x, const arma::SpMat<real>& A, const arma::Col<real>& b, real rtol, real atol, u_long maxit)
{
    auto a = [&A](const arma::Col<real>& x) -> arma::Col<real>
    {
        return A*x;
    };

    auto precond = [](const arma::Col<real>& z) -> arma::Col<real>
    {
        return z;
    };

    return minres(x, a, b, arma::dot<arma::Col<real>,arma::Col<real>>, precond, rtol, atol, maxit);
}

#endif

}
}

#endif