#ifndef NUMERICS_LINEAR_SOLVER_BASE_HPP
#define NUMERICS_LINEAR_SOLVER_BASE_HPP

#include <concepts>

namespace numerics
{
    namespace optimization
    {
        template <std::floating_point real>
        struct LinearSolverResults
        {
            bool success;
            u_long n_iter;
            real residual;

            operator bool() const
            {
                return success;
            }
        };

        class IdentityPreconditioner
        {
        public:
            template <class vec>
            inline vec operator()(const vec& x) const
            {
                return x;
            }
        };
    } // namespace optimization
} // namespace numerics

#endif