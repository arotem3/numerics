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
            template <scalar_field_type scalar>
            void givens_rotation(std::vector<scalar>& h, std::vector<scalar>& cs, std::vector<scalar>& sn, u_long k)
            {
                for (u_long i=0; i < k; ++i)
                {
                    scalar t = cs[i]*h[i] + sn[i]*h[i+1];
                    h[i+1] = -sn[i]*h[i] + cs[i]*h[i+1];
                    h[i] = t;
                }

                precision_t<scalar> t = std::sqrt( std::pow(std::abs(h[k]),2) + std::pow(std::abs(h[k+1]), 2) );
                cs.push_back(h[k] / t);
                sn.push_back(h[k+1] / t);

                h[k] = cs[k]*h[k] + sn[k]*h[k+1];
                h[k+1] = 0;
            }

            template<scalar_field_type scalar>
            bool solve_trimatu(std::vector<scalar>& y, const std::vector<std::vector<scalar>>& H, const std::vector<scalar>& beta)
            {
                long k = H.size();
                
            #if defined(NUMERICS_WITH_ARMA) && !defined(GMRES_NO_ARMA)
                arma::Col<scalar> b(k);
                for (long i=0; i < k; ++i)
                    b[i] = beta[i];

                arma::Mat<scalar> U(k,k);
                for (long j=0; j < k; ++j)
                    for (long i=0; i <= j; ++i)
                        U.at(i,j) = H[j][i];

                bool success = arma::solve(b, arma::trimatu(U), b);
                if (success)
                    y = arma::conv_to<std::vector<scalar>>::from(b);

                return success;
            #else
                // back solve H*y = beta
                y = beta;
                if (std::abs(H[k-1][k-1]) == 0)
                    return false;

                y[k-1] /= H[k-1][k-1];
                for (long i=k-2; i >= 0; --i)
                {
                    for (long j=k-1; j >= i+1; --j)
                        y[i] -= H[j][i] * y[j];

                    if (std::abs(H[i][i]) == 0)
                        return false;

                    y[i] /= H[i][i];
                }

                return true;
            #endif
            }

            template <class Vec, scalar_field_type scalar=typename Vec::value_type>
            bool solve_update(Vec& x, const std::vector<Vec>& Q, const std::vector<std::vector<scalar>>& H, const std::vector<scalar>& beta)
            {
                long k = H.size();
                std::vector<scalar> y;
                bool success = solve_trimatu(y, H, beta);
                if (not success)
                    return false;

                // orthogonal solve Q'*x = y
                for (u_long i=0; i < k; ++i)
                    x += Q[i] * y[i];

                return true;
            }
        } // namespace __gmres

        // solves a general square system of linear equations A*x == b using the
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
                std::vector<std::vector<scalar>> H;

                for (u_long k = 0; k < restart; ++k)
                {
                    it++;

                    // arnoldi iteration
                    Vec q = A(Q.back());
                    q = precond(q);
                    std::vector<scalar> h(k+2);
                    for (u_long i=0; i <= k; ++i)
                    {
                        // h[i] = dot(Q[i], q);
                        h[i] = dot(q, Q[i]); // for compatibility with armadillo which computes dot(x,y) = x'*y instead of y'*x
                        q -= h[i] * Q[i];
                    }
                    h[k+1] = std::sqrt( std::abs(dot(q,q)) );
                    q /= h[k+1];

                    Q.push_back(std::move(q));

                    __gmres::givens_rotation(h, cs, sn, k);
                    H.push_back(std::move(h));

                    beta.push_back(-sn.at(k) * beta.at(k));
                    beta.at(k) = cs.at(k) * beta.at(k);

                    err = std::abs( beta.at(k+1) );
                    res = err;
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
        // solves a general square system of linear equations A*x == b using the
        // preconditioned restarted Generalized Minimum Residual method. This function
        // is a wrapper for gmres for armadillo types. If x is not initialized, it is
        // initialized with zeros.
        template <scalar_field_type scalar, std::invocable<arma::Col<scalar>> Precond>
        inline LinearSolverResults<precision_t<scalar>> gmres(arma::Col<scalar>& x, const arma::Mat<scalar>& A, const arma::Col<scalar>& b, Precond precond, precision_t<scalar> rtol, precision_t<scalar> atol, u_long restart, u_long max_cycles)
        {
            if (not A.is_square())
                throw std::invalid_argument("gmres() error: matrix A is not square.");
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

        // solves a general square system of linear equations A*x == b using the
        // preconditioned restarted Generalized Minimum Residual method. This function
        // is a wrapper for gmres for armadillo types. If x is not initialized, it is
        // initialized with zeros.
        template <scalar_field_type scalar>
        inline LinearSolverResults<precision_t<scalar>> gmres(arma::Col<scalar>& x, const arma::Mat<scalar>& A, const arma::Col<scalar>& b, precision_t<scalar> rtol, precision_t<scalar> atol, u_long restart, u_long max_cycles)
        {
            if (not A.is_square())
                throw std::invalid_argument("gmres() error: matrix A is not square.");
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

        // solves a general square system of linear equations A*x == b using the
        // preconditioned restarted Generalized Minimum Residual method. This function
        // is a wrapper for gmres for armadillo types. If x is not initialized, it is
        // initialized with zeros.
        template <scalar_field_type scalar, std::invocable<arma::Col<scalar>> Precond>
        inline LinearSolverResults<precision_t<scalar>> gmres(arma::Col<scalar>& x, const arma::SpMat<scalar>& A, const arma::Col<scalar>& b, Precond precond, precision_t<scalar> rtol, precision_t<scalar> atol, u_long restart, u_long max_cycles)
        {
            if (not A.is_square())
                throw std::invalid_argument("gmres() error: matrix A is not square.");
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

        // solves a general square system of linear equations A*x == b using the
        // preconditioned restarted Generalized Minimum Residual method. This function
        // is a wrapper for gmres for armadillo types. If x is not initialized, it is
        // initialized with zeros.
        template<scalar_field_type scalar>
        inline LinearSolverResults<precision_t<scalar>> gmres(arma::Col<scalar>& x, const arma::SpMat<scalar>& A, const arma::Col<scalar>& b, precision_t<scalar> rtol, precision_t<scalar> atol, u_long restart, u_long max_cycles)
        {
            if (not A.is_square())
                throw std::invalid_argument("gmres() error: matrix A is not square.");
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