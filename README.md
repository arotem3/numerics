Author: Amit Rotem
Last modified: 04/23/2019

These headers take advantage of the armadillo linear algebra library and require it to compile and function.
All the functions can be found in the header file.

### basic installation instructions:
I have only a simple CMakeLists.txt, so any modifications will have to be provided by you, the user. (sorry...)
1. install [Armadillo +v9.2](http://arma.sourceforge.net/) and [matplotlibcpp](https://github.com/lava/matplotlib-cpp) (optional).
1. `cd /numeric-lib/`
1. `cmake .`
1. `make`
1. `sudo make install`

## numerics.hpp is a numeric library hosting:
* integration (4th order and 7-pt lobatto).

* root finding methods (derivative, derivative free, and mixed methods).
* error control, and approximation options for root finding passed to solver via options struct.

* optimization methods (unconstrained and box constraints. Function, Gradient, and Hessian based methods).

* interpolation schemes (linear, lagrange, cubic, and fourier interpolation).

* data smoothing using thin plate splines and kernel methods.

* simple finite difference methods (for approximating derivatives).
* uniform spectral (fourier) derivatives over an interval.

* machine learning tools (kmeans).

## ODE.hpp is a numerics library for solving ordinary differential equations
* Discrete differentiation matrices
* explicit grid and adaptive IVP solvers (4th order)
* implicit grid and adaptive IVP solvers (1st, 2nd, 5th order)
* implicit solvers use quasi-Newton methods making them more efficient and accurate that fixed point iteration
* event handling and other options passed to solver via options struct

* linear BVP solver (2nd, 4th, spectral order)
* nonlinear BVP solver (2nd, 4th, spectral order)

## statistics.hpp is a statistics (mostly) template library with hypothesis tests, it features:
* basic statistics like mean, median, and variance
* PDFs, CDFs, and quantile functions for the normal, Student's t, and chi-squared distributions
* generic quantile function for user defined cdf, and pdf.
* one and two sample z-tests, t-tests for means.
* proportion test.
* chi squared tests.
* simple resampling/permutation test for means.

There are example codes for every function and class.

Note, many of the examples rely on ["matplotlibcpp.h"](https://github.com/lava/matplotlib-cpp) which is used for visualising results of many of the algorithms. To use this feature install "matplotlib.h" and make sure you have a developer version of python 2.7.

Disclaimer: This library is by no means exceptionally robust or efficient; it does what it can. I implemented many of these programs for fun!
Documentation is now in progress...