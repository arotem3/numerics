#ifndef NUMERICS_OPTIMIZATION_GMRES_HPP
#define NUMERICS_OPTIMIZATION_GMRES_HPP

#include <cmath>
#include <vector>
#include <concepts>

#include "numerics/concepts.hpp"
#include "numerics/optimization/linear_solver_base.hpp"

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

namespace numerics {
    namespace optimization {
        namespace __gmres {
            template <std::floating_point scalar>
            inline scalar _square(scalar x)
            {
                return x*x;
            }

            template <std::floating_point real>
            inline real _square(std::complex<real> x)
            {
                return std::norm(x);
            }

            template <scalar_field_type scalar, typename Iter>
            void givens_rotation(Iter h, std::vector<scalar>& cs, std::vector<scalar>& sn, u_long k)
            {
                auto c = cs.begin();
                auto s = sn.begin();
                Iter hp = h;
                ++hp;
                for (u_long i=0; i < k; ++i, ++c, ++s, ++h, ++hp)
                {
                    scalar t = (*c) * (*h) + (*s) * (*hp);
                    (*hp) = -(*s) * (*h) + (*c) * (*hp);
                    (*h) = t;
                }

                precision_t<scalar> t = std::sqrt( _square(*h) + _square(*hp) );
                cs.push_back((*h) / t);
                sn.push_back((*hp) / t);

                (*h) = cs.back() * (*h) + sn.back() * (*hp);
                (*hp) = scalar(0.0);
            }

            template<typename rIter_A, typename rIter_b>
            bool solve_trimatu(rIter_A a, rIter_b b_rbegin, rIter_b b_rend)
            {
                typedef precision_t<std::remove_reference_t<decltype(*a)>> precision;

                for (rIter_b x = b_rbegin; x != b_rend; ++x)
                {
                    if (std::abs(*a) == precision(0.0))
                        return false;

                    (*x) *= precision(1.0) / (*a);
                    ++a;

                    rIter_b y = x; ++y;
                    for (; y != b_rend; ++y, ++a)
                        (*y) -= (*a) * (*x);
                }

                return true;
            }

            template <typename Vec, typename rIter, scalar_field_type scalar = typename Vec::value_type>
            bool solve_update(Vec& x, const std::vector<Vec>& Q, rIter H, std::vector<scalar>& beta)
            {
                bool success = solve_trimatu(H, beta.rbegin()+1, beta.rend());
                if (not success)
                    return false;

                auto q = Q.begin();
                for (auto y = beta.begin(); y != beta.end(); ++y, ++q)
                    x += (*y) * (*q);

                return true;
            }
        } // namespace __gmres

        // solves a general _square system of linear equations A*x == b using the
        // preconditioned restarted Generalized Minimum Residual method. The vector x
        // should be intialized; any appropriately scaled vector, or the zero vector is
        // likely to converge, however a good initial guess can accelerate convergence.
        // The Vec object should behave like a mathematical vector in a Euclidean vector
        // space, that is, is should have an addition, subtraction, and scalar
        // multiplication operators which return objects that cast back to Vec.
        // Additionally, the inplace addition and subtraction (+=, -=) should also be
        // defined. The LinOp object A should encode the matrix vector product A(x) ->
        // A*x. The Dot object dot should be invocable with two Vec arguments and return
        // the dot product of two vectors, dot(x,y) -> transpose(x)*y. The Precond
        // object precond should be invocable with one Vec argument and return the
        // solution to the preconditioner system: precond(z) -> M\z (where M is a left
        // preconditioner). The parameters rtol and atol specify the relative and
        // absolute tolerance respectively for the solution stopping criteria. For using
        // restarts, restart should be used to specify the number of krylov vectors to
        // store before updating x and restarting the krylov subspace search, and
        // max_cycles should be used to specify the number of outer iterations. Setting
        // restart to the size of x and max_cycles to 1 theoretically produces an exact
        // solution to the system, although propagation of round off errors may require
        // more iteration. This choice requires storing n (=size of x) many krylov
        // vectors which may be too costly for larger problems, for this reason setting
        // restart to a smaller number tends to speed up computation and sometimes
        // accelerates convegence, but restarted gmres is not guaranteed to converge.
        // see:
        // Saad, Y. (2003). Iterative methods for sparse linear systems. Philadelphia:
        // SIAM. 
        template <class Vec, std::invocable<Vec> LinOp, std::invocable<Vec,Vec> Dot, std::invocable<Vec> Precond, scalar_field_type scalar = typename Vec::value_type>
        LinearSolverResults<precision_t<scalar>> gmres(Vec& x, LinOp A, const Vec& b, Dot dot, Precond precond, precision_t<scalar> rtol, precision_t<scalar> atol, u_long restart, u_long max_cycles)
        {
            typedef precision_t<scalar> precision;
            bool success = false;

            precision bnorm = std::sqrt( std::abs(dot(b,b)) );
            precision res;

            u_long it = 0;
            for (u_long i=0; i < max_cycles; ++i)
            {
                Vec r = b - A(x);
                r = precond(r);

                precision rnorm = std::sqrt( std::abs(dot(r,r)) );
                precision err = rnorm;

                std::vector<scalar> sn;
                std::vector<scalar> cs;
                std::vector<scalar> beta = {rnorm};

                std::vector<Vec> Q = {r/rnorm};
                std::vector<scalar> H;

                for (u_long k = 0; k < restart; ++k)
                {
                    it++;

                    // arnoldi iteration
                    Vec q = A(Q.back());
                    q = precond(q);
                    std::vector<scalar> h(k+2);
                    auto hi = h.begin();
                    for (auto qi = Q.begin(); qi != Q.end(); ++qi, ++hi)
                    {
                        (*hi) = dot(q, *qi);
                        q -= (*hi) * (*qi);
                    }
                    (*hi) = sqrt( std::real(dot(q,q)) );
                    q *= precision(1.0) / (*hi);

                    Q.push_back(std::move(q));

                    __gmres::givens_rotation(h.begin(), cs, sn, k);
                    h.pop_back(); // h was a column of an upper Hessenberg matrix, but after givens rotations, it is now a column of an upper triangular matrix, so remove last element.
                    H.insert(H.end(), h.begin(), h.end());

                    beta.push_back(-sn[k] * beta[k]);
                    beta[k] = cs[k] * beta[k];

                    err = std::abs( beta[k+1] );
                    res = err;
                    if (err < rtol*bnorm + atol) {
                        success = true;
                        break;
                    }
                }
                bool solve = __gmres::solve_update(x, Q, H.rbegin(), beta);
                success = success && solve;
                if (success or (not solve))
                    break;
            }

            LinearSolverResults<precision> rslts;
            rslts.success = success;
            rslts.n_iter = it;
            rslts.residual = res;
            return rslts;
        }

        template <class Vec, std::invocable<Vec> LinOp, std::invocable<Vec,Vec> Dot, scalar_field_type scalar = typename Vec::value_type>
        inline LinearSolverResults<precision_t<scalar>> gmres(Vec& x, LinOp A, const Vec& b, Dot dot, precision_t<scalar> rtol, precision_t<scalar> atol, u_long restart, u_long max_cycles)
        {
            return gmres(x, std::forward<LinOp>(A), b, std::forward<Dot>(dot), IdentityPreconditioner{}, rtol, atol, restart, max_cycles);
        }

        #ifdef NUMERICS_WITH_ARMA
        // solves a general _square system of linear equations A*x == b using the
        // preconditioned restarted Generalized Minimum Residual method. This function
        // is a wrapper for gmres for armadillo types. If x is not initialized, it is
        // initialized with zeros.
        template <scalar_field_type scalar, std::invocable<arma::Col<scalar>> Precond>
        inline LinearSolverResults<precision_t<scalar>> gmres(arma::Col<scalar>& x, const arma::Mat<scalar>& A, const arma::Col<scalar>& b, Precond precond, precision_t<scalar> rtol, precision_t<scalar> atol, u_long restart, u_long max_cycles)
        {
            if (not A.is_square())
                throw std::invalid_argument("gmres() error: matrix A is not _square.");
            if (A.n_rows != b.n_rows)
                throw std::invalid_argument("gmres() error: A.n_rows (=" + std::to_string(A.n_rows) + ") != b.n_rows (=" + std::to_string(b.n_rows) + ")");
            if (x.n_cols != A.n_cols)
                x = arma::zeros<arma::Col<scalar>>(A.n_cols);
            if (max_cycles == 0)
                max_cycles = x.n_elem;
            
            auto a = [&A](const arma::Col<scalar>& z) -> arma::Col<scalar>
            {
                return A*z;
            };

            return gmres(x, a, b, arma::cdot<arma::Col<scalar>,arma::Col<scalar>>, std::forward<Precond>(precond), rtol, atol, restart, max_cycles);
        }

        // solves a general _square system of linear equations A*x == b using the
        // preconditioned restarted Generalized Minimum Residual method. This function
        // is a wrapper for gmres for armadillo types. If x is not initialized, it is
        // initialized with zeros.
        template <scalar_field_type scalar>
        inline LinearSolverResults<precision_t<scalar>> gmres(arma::Col<scalar>& x, const arma::Mat<scalar>& A, const arma::Col<scalar>& b, precision_t<scalar> rtol, precision_t<scalar> atol, u_long restart, u_long max_cycles)
        {
            if (not A.is_square())
                throw std::invalid_argument("gmres() error: matrix A is not _square.");
            if (A.n_rows != b.n_rows)
                throw std::invalid_argument("gmres() error: A.n_rows (=" + std::to_string(A.n_rows) + ") != b.n_rows (=" + std::to_string(b.n_rows) + ")");
            if (x.n_cols != A.n_cols)
                x = arma::zeros<arma::Col<scalar>>(A.n_cols);
            if (max_cycles == 0)
                max_cycles = x.n_elem;
            
            auto a = [&A](const arma::Col<scalar>& z) -> arma::Col<scalar>
            {
                return A*z;
            };
            return gmres(x, a, b, arma::cdot<arma::Col<scalar>,arma::Col<scalar>>, IdentityPreconditioner{}, rtol, atol, restart, max_cycles);
        }

        // solves a general _square system of linear equations A*x == b using the
        // preconditioned restarted Generalized Minimum Residual method. This function
        // is a wrapper for gmres for armadillo types. If x is not initialized, it is
        // initialized with zeros.
        template <scalar_field_type scalar, std::invocable<arma::Col<scalar>> Precond>
        inline LinearSolverResults<precision_t<scalar>> gmres(arma::Col<scalar>& x, const arma::SpMat<scalar>& A, const arma::Col<scalar>& b, Precond precond, precision_t<scalar> rtol, precision_t<scalar> atol, u_long restart, u_long max_cycles)
        {
            if (not A.is_square())
                throw std::invalid_argument("gmres() error: matrix A is not _square.");
            if (A.n_rows != b.n_rows)
                throw std::invalid_argument("gmres() error: A.n_rows (=" + std::to_string(A.n_rows) + ") != b.n_rows (=" + std::to_string(b.n_rows) + ")");
            if (x.n_cols != A.n_cols)
                x = arma::zeros<arma::Col<scalar>>(A.n_cols);
            if (max_cycles == 0)
                max_cycles = x.n_elem;

            auto a = [&A](const arma::Col<scalar>& z) -> arma::Col<scalar>
            {
                return A*z;
            };
            return gmres(x, a, b, arma::cdot<arma::Col<scalar>,arma::Col<scalar>>, std::forward<Precond>(precond), rtol, atol, restart, max_cycles);
        }

        // solves a general _square system of linear equations A*x == b using the
        // preconditioned restarted Generalized Minimum Residual method. This function
        // is a wrapper for gmres for armadillo types. If x is not initialized, it is
        // initialized with zeros.
        template<scalar_field_type scalar>
        inline LinearSolverResults<precision_t<scalar>> gmres(arma::Col<scalar>& x, const arma::SpMat<scalar>& A, const arma::Col<scalar>& b, precision_t<scalar> rtol, precision_t<scalar> atol, u_long restart, u_long max_cycles)
        {
            if (not A.is_square())
                throw std::invalid_argument("gmres() error: matrix A is not _square.");
            if (A.n_rows != b.n_rows)
                throw std::invalid_argument("gmres() error: A.n_rows (=" + std::to_string(A.n_rows) + ") != b.n_rows (=" + std::to_string(b.n_rows) + ")");
            if (x.n_cols != A.n_cols)
                x = arma::zeros<arma::Col<scalar>>(A.n_cols);
            if (max_cycles == 0)
                max_cycles = x.n_elem;
            
            
            auto a = [&A](const arma::Col<scalar>& z) -> arma::Col<scalar>
            {
                return A*z;
            };
            return gmres(x, a, b, arma::cdot<arma::Col<scalar>,arma::Col<scalar>>, IdentityPreconditioner{}, rtol, atol, restart, max_cycles);
        }
        #endif
    } // namespace optimization
} // namespace numerics
#endif