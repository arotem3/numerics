#ifndef NUMERICS_INTERPOLATION_LAGRANGE_INTERP_HPP
#define NUMERICS_INTERPOLATION_LAGRANGE_INTERP_HPP

namespace numerics
{
    // sets yy to the polynomial interpolation of the data specified by
    // iterators x_first and y_first (to x_last, exclusive) at point xx. The
    // iterpolated values in y_first .. can be floating point, complex, valarray
    // or any other type with mathematical vector operations, i.e. addition via
    // `+` and `+=`, and (right) scalar multiplcation via `*`. The variable yy
    // should be initialized to zero, and, if it is a vector, should have the
    // same size as the data in y_first so that the + operations make sense.
    template <typename real, typename vec, typename real_it, typename vec_it>
    void lagrange_interp(real_it x_first, real_it x_last, vec_it y_first, real xx, vec& yy)
    {
        for (real_it xi = x_first; xi != x_last; ++xi, ++y_first)
        {
            real p = 1;
            for (real_it xj = x_first; xj != x_last; ++xj)
            {
                if (xi != xj)
                    p *= (xx - *xj) / (*xi - *xj);
            }
            yy += p * (*y_first);
        }
    }
} // namespace numerics

#endif