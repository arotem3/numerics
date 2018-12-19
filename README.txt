Author: Amit Rotem
Last modified: 12/17/2018

These libraries take advantage of the armadillo linear algebra library and require it to compile and function.
All the functions can be found in the header file.

numerics.hpp is a numeric library hosting:
    *) integration (2nd, 4th, 7-pt lobatto order).

    *) root finding methods (derivative, derivative free, and mixed methods).
    *) error control, and approximation options for root finding passed to solver via options struct.

    *) optimization methods (unconstrained and box constraint. Function, Gradient, and Hessian based methods).

    *) interpolation schemes (linear, lagrange, cubic, and fourier interpolation).

    *) simple finite difference methods (for approximating derivatives).
    *) uniform spectral (fourier) derivatives over an interval.

    *) machine learning tools (kmeans).

ODE.hpp is a numerics library for solving ordinary differential equations
    *) explicit grid and adaptive IVP solvers (4th order)
    *) implicit grid and adaptive IVP solvers (1st, 2nd, 5th order)
    *) implicit solvers use Newton like methods making them more efficient and accurate that fixed point iteration
    *) event handling and other options passed to solver via options struct

    *) linear BVP solver (2nd, 4th, spectral order)
    *) nonlinear BVP solver (spectral order)

statistics.hpp is a statistics (mostly) template library with hypothesis tests, it features:
    *) basic statistics like mean, median, and variance
    *) PDFs, CDFs, and quantile functions for the normal, Student's t, and chi-squared distributions
    *) generic quantile function for user defined cdf, and pdf.
    *) one and two sample z-tests, t-tests for means.
    *) proportion test.
    *) chi squared tests.
    *) simple resampling/permutation test for means.

vector_operators.hpp is a template library with basic elementwise functions:
    *) +, -, *, / operators (and their += counterparts)
    *) &&, || operators
    *) ==, !=, <, <=, >, >= operators
    *) any() all() functions
    *) most cmath functions (the ones I find useful)
    *) sum() and norm() functions

There are example codes for every function and class.

Disclaimer: I do not take credit for any algorithm, I just implemented them in C++ for fun...
This library is by no means exceptionally robust or efficient; it does what it can.
Documentation is really only in the form of examples provided in the 'examples' directory.