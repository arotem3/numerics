cmake_minimum_required(VERSION 3.10)
project(numerics)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)

add_library(numerics SHARED
        include/numerics.hpp
        include/numerics/data_science.hpp
        include/numerics/derivatives.hpp
        include/numerics/integrals.hpp
        include/numerics/interpolation.hpp
        include/numerics/ode.hpp
        include/numerics/optimization.hpp
        include/numerics/utility.hpp

        src/data_science/k_fold.cpp
        src/data_science/kernel_smooth.cpp
        src/data_science/kmeans.cpp
        src/data_science/regularizer.cpp
        src/data_science/splines.cpp
        src/data_science/logistic_regression.cpp

        src/derivatives/approx_jacobian.cpp
        src/derivatives/deriv.cpp
        src/derivatives/grad.cpp
        src/derivatives/spectral_deriv.cpp

        src/integrals/chebyshev_integral.cpp
        src/integrals/lobatto_integral.cpp
        src/integrals/simpson_integral.cpp

        src/interpolation/cubic_interp.cpp
        src/interpolation/lagrange_interp.cpp
        src/interpolation/poly_interp.cpp
        src/interpolation/sinc_interp.cpp
        src/interpolation/polynomials.cpp

        src/ode/am1.cpp
        src/ode/am2.cpp
        src/ode/bvp.cpp
        src/ode/cheb.cpp
        src/ode/diffmat.cpp
        src/ode/ivp_events.cpp
        src/ode/rk4.cpp
        src/ode/rk5i.cpp
        src/ode/rk45.cpp
        src/ode/poisson2d.cpp

        src/optimization/adj_gd.cpp
        src/optimization/bfgs.cpp
        src/optimization/broyd.cpp
        src/optimization/cgd.cpp
        src/optimization/fzero.cpp
        src/optimization/gen_optim.cpp
        src/optimization/lbfgs.cpp
        src/optimization/line_min.cpp
        src/optimization/lmlsqr.cpp
        src/optimization/mgd.cpp
        src/optimization/mix_fpi.cpp
        src/optimization/nelder_mead.cpp
        src/optimization/newton.cpp
        src/optimization/simplex.cpp
        src/optimization/wolfe_step.cpp
        
        src/utility/cyc_queue.cpp
        src/utility/meshgrid.cpp
        src/utility/sample_from.cpp
)

install(TARGETS numerics DESTINATION lib)
install(FILES include/numerics.hpp DESTINATION include)
install(DIRECTORY include/numerics DESTINATION include)