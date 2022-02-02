#ifndef NUMERICS_OPTIMIZATION_MINRES_HPP
#define NUMERICS_OPTIMIZATION_MINRES_HPP

#include <limits>
#include <cmath>
#include "numerics/optimization/linear_solver_base.hpp"

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

namespace numerics {
    namespace optimization {
        // solves the symmetric linear system A*x == b using the preconditioned minimum
        // residual method. The vector x should be intialized; any appropriately scaled
        // vector, or the zero vector is likely to converge, however a good initial
        // guess can accelerate convergence. The Vec object should behave like a
        // mathematical vector in a Euclidean vector space, that is, is should have an
        // addition, subtraction, and scalar multiplication operators which return
        // objects that cast back to Vec. Additionally, the inplace addition and
        // subtraction (+=, -=) should also be defined. The LinOp object A should encode
        // the matrix vector product A(x) -> A*x. The Dot object dot should be invocable
        // with two Vec arguments and return the dot product of two vectors, dot(x,y) ->
        // transpose(x)*y. The Precond object precond should be invocable with one Vec
        // argument and return the solution to the preconditioner system: precond(z) ->
        // M\z (where M is a left preconditioner and should be symmetric positive
        // definite). The parameters rtol and atol specify the relative and absolute
        // tolerance respectively for the solution stopping criteria. The parameter
        // maxit specifies the maximum iterations to compute. At every iteration, minres
        // decreases the residual ||b - A*x||.
        // this code translates the matlab code here:
        // https://web.stanford.edu/group/SOL/software/minres/
        // see:
        // C. C. Paige and M. A. Saunders (1975). Solution of sparse indefinite systems
        // of linear equations, SIAM J. Numerical Analysis 12, 617-629.
        template <class Vec, std::invocable<Vec> LinOp, std::invocable<Vec,Vec> Dot, std::invocable<Vec> Precond, std::floating_point real=typename Vec::value_type>
        LinearSolverResults<real> minres(Vec& x, LinOp A, const Vec& b, Dot dot, Precond precond, real rtol, real atol, u_long maxit)
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

            u_long i;
            for (i=1; i <= maxit; ++i)
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

            LinearSolverResults<real> rslts;
            rslts.success = success;
            rslts.n_iter = i;
            rslts.residual = phibar;
            return rslts;
        }

        template <class Vec, std::invocable<Vec> LinOp, std::invocable<Vec,Vec> Dot, std::floating_point real=typename Vec::value_type>
        inline LinearSolverResults<real> minres(Vec& x, LinOp A, const Vec& b, Dot dot, real rtol, real atol, u_long maxit)
        {
            return minres(x, std::forward<LinOp>(A), b, std::forward<Dot>(dot), IdentityPreconditioner{}, rtol, atol, maxit);
        }

        #ifdef NUMERICS_WITH_ARMA
        // solves A*x == b using the minimum residual method. This method is a
        // specialization of minres() to armadillo types. For this method, if x is not
        // initialized, it will be filled with zeros.
        template <std::floating_point real, class Precond>
        inline LinearSolverResults<real> minres(arma::Col<real>& x, const arma::Mat<real>& A, const arma::Col<real>& b, Precond precond, real rtol, real atol, u_long maxit)
        {
            if (not A.is_symmetric())
                throw std::invalid_argument("minres() error: matrix A is not symmetric.");
            if (A.n_rows != b.n_rows)
                throw std::invalid_argument("minres() error: A.n_rows (=" + std::to_string(A.n_rows) + ") != b.n_rows (=" + std::to_string(b.n_rows) + ")");
            if (x.n_cols != A.n_cols)
                x = arma::zeros<arma::Col<real>>(A.n_cols);

            auto a = [&A](const arma::Col<real>& x) -> arma::Col<real>
            {
                return A*x;
            };

            return minres(x, a, b, arma::dot<arma::Col<real>,arma::Col<real>>, std::forward<Precond>(precond), rtol, atol, maxit);
        }

        // solves A*x == b using the minimum residual method. This method is a
        // specialization of minres() to armadillo types. For this method, if x is not
        // initialized, it will be filled with zeros.
        template <std::floating_point real>
        inline LinearSolverResults<real> minres(arma::Col<real>& x, const arma::Mat<real>& A, const arma::Col<real>& b, real rtol, real atol, u_long maxit)
        {
            if (not A.is_symmetric())
                throw std::invalid_argument("minres() error: matrix A is not symmetric.");
            if (A.n_rows != b.n_rows)
                throw std::invalid_argument("minres() error: A.n_rows (=" + std::to_string(A.n_rows) + ") != b.n_rows (=" + std::to_string(b.n_rows) + ")");
            if (x.n_cols != A.n_cols)
                x = arma::zeros<arma::Col<real>>(A.n_cols);
            
            auto a = [&A](const arma::Col<real>& x) -> arma::Col<real>
            {
                return A*x;
            };
            return minres(x, a, b, arma::dot<arma::Col<real>,arma::Col<real>>, IdentityPreconditioner{}, rtol, atol, maxit);
        }

        // solves A*x == b using the minimum residual method. This method is a
        // specialization of minres() to armadillo types. For this method, if x is not
        // initialized, it will be filled with zeros.
        template <std::floating_point real, class Precond>
        inline LinearSolverResults<real> minres(arma::Col<real>& x, const arma::SpMat<real>& A, const arma::Col<real>& b, Precond precond, real rtol, real atol, u_long maxit)
        {
            if (not A.is_symmetric())
                throw std::invalid_argument("minres() error: matrix A is not symmetric.");
            if (A.n_rows != b.n_rows)
                throw std::invalid_argument("minres() error: A.n_rows (=" + std::to_string(A.n_rows) + ") != b.n_rows (=" + std::to_string(b.n_rows) + ")");
            if (x.n_cols != A.n_cols)
                x = arma::zeros<arma::Col<real>>(A.n_cols);
            
            auto a = [&A](const arma::Col<real>& x) -> arma::Col<real>
            {
                return A*x;
            };

            return minres(x, a, b, arma::dot<arma::Col<real>,arma::Col<real>>, std::forward<Precond>(precond), rtol, atol, maxit);
        }

        // solves A*x == b using the minimum residual method. This method is a
        // specialization of minres() to armadillo types. For this method, if x is not
        // initialized, it will be filled with zeros.
        template <std::floating_point real>
        inline LinearSolverResults<real> minres(arma::Col<real>& x, const arma::SpMat<real>& A, const arma::Col<real>& b, real rtol, real atol, u_long maxit)
        {
            if (not A.is_symmetric())
                throw std::invalid_argument("minres() error: matrix A is not symmetric.");
            if (A.n_rows != b.n_rows)
                throw std::invalid_argument("minres() error: A.n_rows (=" + std::to_string(A.n_rows) + ") != b.n_rows (=" + std::to_string(b.n_rows) + ")");
            if (x.n_cols != A.n_cols)
                x = arma::zeros<arma::Col<real>>(A.n_cols);

            auto a = [&A](const arma::Col<real>& x) -> arma::Col<real>
            {
                return A*x;
            };
            return minres(x, a, b, arma::dot<arma::Col<real>,arma::Col<real>>, IdentityPreconditioner{}, rtol, atol, maxit);
        }

        #endif
    } // namespace optimization
} // namespace numerics

#endif