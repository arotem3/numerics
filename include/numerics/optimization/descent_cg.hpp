#ifndef NUMERICS_OPTIMIZATION_DESCENT_CG
#define NUMERICS_OPTIMIZATION_DESCENT_CG

#include <cmath>
#include <utility>
#include <concepts>

#if defined(ARMA_INCLUDES) && !defined(NUMERICS_WITH_ARMA)
#define NUMERICS_WITH_ARMA
#endif

namespace numerics {
    namespace optimization {
        // uses conjugate gradient to find an approximate newton direction s such that
        // H*s = g where H is the hessian matrix (maybe indefinite), g is the NEGATIVE
        // gradient. The solver stops if dot(p, H(p)) < 0 (where p is the cg update) or
        // a convergence criteria is met. The vector s should be initialized.
        // see:
        // (2006) Large-Scale Unconstrained Optimization. In: Numerical Optimization.
        // Springer Series in Operations Research and Financial Engineering. Springer,
        // New York, NY. https://doi-org.proxy2.cl.msu.edu/10.1007/978-0-387-40065-5_7
        template <class Vec, std::invocable<Vec> LinOp, std::invocable<Vec,Vec> Dot, std::invocable<Vec> Precond, std::floating_point real = typename Vec::value_type>
        bool descent_cg(Vec& s, LinOp H, const Vec& g, Dot dot, Precond precond, u_long maxit)
        {
            real gnorm = std::sqrt(dot(g,g));
            real tol = std::min<real>(0.5f, std::sqrt(gnorm))*gnorm; // for superlinear convergence of newton-methods
            
            Vec r = g - H(s);

            bool success = false;
            Vec p;
            real rho_prev, rho;
            for (u_long i=0; i < maxit; ++i)
            {
                Vec z = precond(r);
                rho = dot(r, z);

                if (i == 0)
                    p = std::move(z);
                else {
                    real beta = rho / rho_prev;
                    p = z + beta * p;
                }

                Vec Hp = H(p);
                real pHp = dot(Hp,p);
                real alpha = rho / pHp;
                
                if (pHp <= 0) {
                    if (i == 0)
                        s = g;
                    success = true;
                    break;
                }

                s += alpha * p;
                r -= alpha * Hp;

                rho_prev = rho;

                if (std::sqrt(dot(r,r)) < tol) {
                    success = true;
                    break;
                }
            }

            return success;
        }

        template <class Vec, std::invocable<Vec> LinOp, std::invocable<Vec,Vec> Dot, std::floating_point real = typename Vec::value_type>
        inline bool descent_cg(Vec& s, LinOp H, const Vec& g, Dot dot, u_long max_iter)
        {
            auto precond = [](const Vec& z) -> Vec
            {
                return z;
            };
            return descent_cg(s, std::forward<LinOp>(H), g, std::forward<Dot>(dot), precond, max_iter);
        }

        #ifdef NUMERICS_WITH_ARMA
        template <std::floating_point real>
        inline bool descent_cg(arma::Col<real>& s, const arma::Mat<real>& H, const arma::Col<real>& g, u_long max_iter)
        {
            if (not H.is_square())
                throw std::invalid_argument("descent_cg() error: matrix not square!");
            if (H.n_rows != g.n_rows)
                throw std::invalid_argument("descent_cg() error: H.n_rows (=" + std::to_string(H.n_rows) + ") != g.n_rows (=" + std::to_string(g.n_rows) + ")");
            if (s.n_elem != g.n_elem)
                s = arma::zeros<arma::Col<real>>(g.n_elem);
            if (max_iter == 0)
                max_iter = g.n_elem;
            
            auto Hprod = [&H](const arma::Col<real>& z) -> arma::Col<real>
            {
                return H*z;
            };
            auto precond = [](const arma::Col<real>& z) -> arma::Col<real>
            {
                return z;
            };
            return descent_cg(s, Hprod, g, arma::dot<arma::Col<real>,arma::Col<real>>, precond, max_iter);
        }

        template <std::floating_point real, class Precond>
        inline bool descent_cg(arma::Col<real>& s, const arma::Mat<real>& H, const arma::Col<real>& g, Precond precond, u_long max_iter)
        {
            if (not H.is_square())
                throw std::invalid_argument("descent_cg() error: matrix not square!");
            if (H.n_rows != g.n_rows)
                throw std::invalid_argument("descent_cg() error: H.n_rows (=" + std::to_string(H.n_rows) + ") != g.n_rows (=" + std::to_string(g.n_rows) + ")");
            if (s.n_elem != g.n_elem)
                s = arma::zeros<arma::Col<real>>(g.n_elem);
            if (max_iter == 0)
                max_iter = g.n_elem;
            
            auto Hprod = [&H](const arma::Col<real>& z) -> arma::Col<real>
            {
                return H*z;
            };
            return descent_cg(s, Hprod, g, arma::dot<arma::Col<real>,arma::Col<real>>, std::forward<Precond>(precond), max_iter);
        }

        template <std::floating_point real>
        inline bool descent_cg(arma::Col<real>& s, const arma::SpMat<real>& H, const arma::Col<real>& g, u_long max_iter)
        {
            if (not H.is_square())
                throw std::invalid_argument("descent_cg() error: matrix not square!");
            if (H.n_rows != g.n_rows)
                throw std::invalid_argument("descent_cg() error: H.n_rows (=" + std::to_string(H.n_rows) + ") != g.n_rows (=" + std::to_string(g.n_rows) + ")");
            if (s.n_elem != g.n_elem)
                s = arma::zeros<arma::Col<real>>(g.n_elem);
            if (max_iter == 0)
                max_iter = g.n_elem;
            
            auto Hprod = [&H](const arma::Col<real>& z) -> arma::Col<real>
            {
                return H*z;
            };
            auto precond = [](const arma::Col<real>& z) -> arma::Col<real>
            {
                return z;
            };
            return descent_cg(s, Hprod, g, arma::dot<arma::Col<real>,arma::Col<real>>, precond, max_iter);
        }

        template <std::floating_point real, class Precond>
        inline bool descent_cg(arma::Col<real>& s, const arma::SpMat<real>& H, const arma::Col<real>& g, Precond precond, u_long max_iter)
        {
            if (not H.is_square())
                throw std::invalid_argument("descent_cg() error: matrix not square!");
            if (H.n_rows != g.n_rows)
                throw std::invalid_argument("descent_cg() error: H.n_rows (=" + std::to_string(H.n_rows) + ") != g.n_rows (=" + std::to_string(g.n_rows) + ")");
            if (s.n_elem != g.n_elem)
                s = arma::zeros<arma::Col<real>>(g.n_elem);
            if (max_iter == 0)
                max_iter = g.n_elem;
            
            auto Hprod = [&H](const arma::Col<real>& z) -> arma::Col<real>
            {
                return H*z;
            };
            return descent_cg(s, Hprod, g, arma::dot<arma::Col<real>,arma::Col<real>>, std::forward<Precond>(precond), max_iter);
        }
        #endif
    } // namespace optimization
} // namespace numerics
#endif