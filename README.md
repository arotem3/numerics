Author: Amit Rotem

Last modified: 10/24/2019

These headers take advantage of the armadillo linear algebra library and require it for both compiling and linking.
All the functions can be found in the header file and in the documentation file.

### basic installation instructions:
I have only a simple CMakeLists.txt, so any modifications will have to be provided by you, the user. This library has only been tested on linux. Most recent versions of gcc should be able to compile OpenMP instructions, this library uses some OpenMP instructions here and there, so verifying `omp.h` exists on your system will significantly improve the performance of some of the data science functions.
1. install [Armadillo +v9.2](http://arma.sourceforge.net/) and [matplotlibcpp](https://github.com/lava/matplotlib-cpp) (optional).
1. `cd /numeric-lib/`
1. `cmake .`
1. `make`
1. `sudo make install`

### basic compiling instructions:
This assumes you have both armadillo and numerics installed in the default locations that your compiler can find (e.g. `/usr/local/lib/`).
```
g++ main.cpp -O3 -lnumerics -larmadillo
```
The order is important because `libnumerics` also has to link against `libarmadillo`. It is also recommended that you use optimization during compile, such as `-O2`, or `-O3`.

### `numerics.hpp` is a scientific computing library hosting:
* integration (4th order, 7-pt lobatto, spectral).

* root finding methods (derivative, derivative free, and mixed methods).
* error control, and approximation options for root finding passed to solver via class interface.

* optimization methods (unconstrained and box constraints. Function, Gradient, and Hessian based methods).

* interpolation schemes (lagrange, cubic, and fourier interpolation).

* machine learning and data analysis tools:
    * train/test splitting.
    * k-means clusttering.
    * regularized linear regression.
    * kernel linear basis regression and classification.
    * kernel based smoothing.
    * kernel density estimation.
    * nearest neighbors regression and classification.
    * built in cross validation for automatic parameter selection for all models.

* simple finite difference methods (for approximating derivatives).
* uniform spectral derivatives over an interval.

### namespace `numerics::ode` for solving ordinary differential equations
* Discrete differentiation matrices
* explicit constant and adaptive step size IVP solvers (4th order)
* implicit constant and adaptive step size IVP solvers (1st, 2nd, 4th, and 5th order)
* implicit solvers use quasi-Newton methods making them more efficient than fixed point iteration.
* event handling and other options passed to solver via class interface.
* nonlinear BVP solver (4th, and upto spectral order)

For an in depth overview, refer to the `documentation.md` file. There are also example files for most of the functions found in `/examples/` directory.

Note, many of the examples rely on ["matplotlibcpp.h"](https://github.com/lava/matplotlib-cpp) which is used for visualising results of many of the algorithms. To use this feature install "matplotlibcpp.h" and make sure you have a developer version of python 2.7.